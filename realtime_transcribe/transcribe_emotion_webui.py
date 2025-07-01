import gradio as gr
import base64
import requests
import soundfile as sf

import base64
import io

import queue
import threading
import os
import numpy as np
import json
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def numpy_to_base64_wav(data, sr):
    with io.BytesIO() as wav_io:
        sf.write(wav_io, data, sr, format="wav")
        wav_io.seek(0)
        base64_audio = base64.b64encode(wav_io.read()).decode("utf-8")
    return base64_audio


def base64_to_numpy_audio(audio_base64):
    if not audio_base64:
        return None
    audio_bytes = base64.b64decode(audio_base64)
    with io.BytesIO(audio_bytes) as audio_io:
        data, samplerate = sf.read(audio_io, dtype='float32')
    return samplerate, data


class ProcessorResult:
    def __init__(self, input_audio_buffer_size: int, text: str):
        self.input_audio_buffer_size = input_audio_buffer_size
        self.text = text


class TranscribedChunk:
    def __init__(self, text: str, is_fixed: bool = False):
        self.text = text
        self.is_fixed = is_fixed

    def __repr__(self):
        return f"{self.text}[{'fixed' if self.is_fixed else 'active'}]"


class RealtimeVoiceProcessor:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        self.result_queue.put(ProcessorResult(0, "INIT"))

        # Initialize Whisper model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "distil-whisper/distil-large-v3.5"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, cache_dir="cache"
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Initialize Silero VAD
        self.silero_model, self.silero_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.silero_model.to("cpu")
        (self.get_speech_timestamps, _, self.read_audio, _, _) = self.silero_utils

        self.full_running_transcription = []  # List of TranscribedChunk objects

    def get_partial_text_context(self, remove_last_n_words=3):
        if not self.full_running_transcription:
            return "", "", ""

        # Get all text from chunks
        full_text_without_last_chunk = "".join(chunk.text for chunk in self.full_running_transcription[-4:-1])

        if self.full_running_transcription[-1].is_fixed is True:
            return full_text_without_last_chunk, "", ""

        last_chunk_words = self.full_running_transcription[-1].text.strip().split()

        partial_text = " ".join(last_chunk_words[:-remove_last_n_words])
        removed_words = " ".join(last_chunk_words[-remove_last_n_words:])
        full_partial_text = full_text_without_last_chunk + partial_text

        return full_partial_text, partial_text, removed_words

    def get_full_text(self):
        return "".join(chunk.text for chunk in self.full_running_transcription)

    def vad_step(self, audio_data):
        # Convert audio data to tensor for Silero VAD
        data = torch.tensor(audio_data, dtype=torch.float32)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(data, self.silero_model, return_seconds=True)
        print(speech_timestamps)
        
        # If no speech detected, return None
        if not speech_timestamps:
            return None
        
        # Find first start and last end timestamps
        first_start = speech_timestamps[0]["start"]
        last_end = speech_timestamps[-1]["end"]
        
        # Convert timestamps to sample indices (assuming 16kHz sample rate)
        sample_rate = 16000
        start_sample = int(first_start * sample_rate)
        end_sample = int(last_end * sample_rate)
        
        # Trim audio to speech segments
        trimmed_audio = audio_data[start_sample:end_sample]
        
        return trimmed_audio

    def predict_emotion(self):
        full_text = self.get_full_text().strip()
        
        if not full_text:
            return "neutral"
        
        # Create messages for LLM emotion analysis
        messages = [
            {
                "role": "system",
                "content": "You are an emotion analysis expert. Analyze the given text and respond with only one word from these emotions: <happy>, <sad>, <angry>, <fear>, <surprise>, <disgust>, <neutral>."
            },
            {
                "role": "user", 
                "content": f"Analyze the emotion in this text: '{full_text}'"
            },
        ]
        
        try:
            # Send request to local LLM server
            response = requests.post(
                "http://localhost:8069/chat/completions",
                json={
                    "messages": messages,
                    "stream": False,
                    "max_tokens": 10,
                    "temperature": 0.1
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                emotion = result["choices"][0]["message"]["content"].strip().lower()
                
                # Validate emotion is in expected list
                valid_emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
                if emotion in valid_emotions:
                    return emotion
                else:
                    # Fallback to finding emotion in response
                    for valid_emotion in valid_emotions:
                        if valid_emotion in emotion:
                            return valid_emotion
                    return "neutral"
            else:
                return "neutral"
                
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return "neutral"

    def post_process_transcription(self):
        # Joint all fixed chunks into a single string
        fixed_chunks = [chunk for chunk in self.full_running_transcription if chunk.is_fixed]
        joined_text = "".join(chunk.text for chunk in fixed_chunks)
        self.full_running_transcription = [TranscribedChunk(joined_text, is_fixed=True)] + [chunk for chunk in self.full_running_transcription if not chunk.is_fixed]

    def transcribe_audio(self, base64_audio):
        # Convert base64 audio back to numpy array
        sr, audio_data = base64_to_numpy_audio(base64_audio)

        target_sr = 16000
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)

        audio_data = audio_data[-target_sr * 16:]

        audio_data = self.vad_step(audio_data)
        if audio_data is None:
            return "..."

        # Get partial text context from existing transcription
        full_partial_text, part_partial_text, removed_words = self.get_partial_text_context(remove_last_n_words=8)
        partial_text = full_partial_text

        # Prepare inputs for Whisper
        inputs = self.processor(audio_data, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device).to(self.torch_dtype) if v.dtype == torch.float32 else v.to(self.device)
                  for k, v in inputs.items()}

        # Create proper decoder input with Whisper special tokens
        decoder_start_token_id = self.model.config.decoder_start_token_id
        lang_token = self.processor.tokenizer.convert_tokens_to_ids("<|en|>")
        task_token = self.processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_token = self.processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

        prompt_tokens = self.processor.tokenizer.encode(partial_text, add_special_tokens=False) if partial_text else []
        decoder_input_ids = [decoder_start_token_id, lang_token, task_token, notimestamps_token] + prompt_tokens
        decoder_input_ids = torch.tensor([decoder_input_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )

        generated_text = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if not self.full_running_transcription:
            self.full_running_transcription.append(
                TranscribedChunk(generated_text[len(removed_words):], is_fixed=False))
        elif len(generated_text.split()) <= 2:
            # Remove silence
            self.full_running_transcription[-1].is_fixed = True
        elif removed_words and generated_text.strip().startswith(removed_words.strip()):
            if generated_text.startswith(" "):
                removed_words = " " + removed_words.strip()

            self.full_running_transcription[-1].is_fixed = True
            self.full_running_transcription.append(
                TranscribedChunk(generated_text[len(removed_words):], is_fixed=False))
        else:
            # Remove repetition
            is_repetition = False
            for chunk in self.full_running_transcription[-2:]:
                if chunk.text.lower().strip() == generated_text.lower().strip():
                    is_repetition = True
            if not is_repetition:
                self.full_running_transcription[-1].is_fixed = False
                self.full_running_transcription[-1].text = part_partial_text + generated_text

        predicted_emotion = self.predict_emotion()

        print(f"Partial context: '{partial_text}'")
        print(f"Removed words: '{removed_words}'")
        print(f"Generated: '{generated_text}'")
        print(f"Full transcription: '{''.join([str(_) for _ in self.full_running_transcription])}'")
        
        # self.post_process_transcription()

        return (
            f"# EMOTION: {predicted_emotion.upper()}\n"
            f"## {self.get_full_text()}  \n"
            f"**Partial context:** `{partial_text}`  \n"
            f"**Removed words:** `{removed_words}`  \n"
            f"**Generated:** `{generated_text}`  \n"
            f"**Chunks:** `{''.join([str(_) for _ in self.full_running_transcription])}`"
        )

    def run(self):
        while True:
            sr, data = self.task_queue.get()
            input_audio_buffer_size = len(data)

            if not np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float32)
                data /= 32768.0

            base64_audio = numpy_to_base64_wav(data, sr)
            transcribed_text = self.transcribe_audio(base64_audio=base64_audio)

            self.result_queue.put(ProcessorResult(input_audio_buffer_size=0, text=transcribed_text))


def process_audio_streaming(audio_buffer, new_chunk, previous_text_display):
    sr, y = new_chunk
    if audio_buffer is not None:
        audio_buffer = np.concatenate([audio_buffer, y])
    else:
        audio_buffer = y

    try:
        if len(audio_buffer) < sr * 1:
            raise queue.Empty

        processed_result = realtime_audio_processor.result_queue.get_nowait()

        # Trim audio_buffer to only include new audio since last turn
        input_audio_buffer_size = processed_result.input_audio_buffer_size
        # Update audio_buffer to remove processed part for next iteration's state
        audio_buffer = audio_buffer[-32 * sr:]

        realtime_audio_processor.task_queue.put((sr, audio_buffer))

        return audio_buffer, processed_result.text
    except queue.Empty:
        return audio_buffer, previous_text_display


# Chatbot instance
realtime_audio_processor = RealtimeVoiceProcessor()

with gr.Blocks(title="Assistant") as demo:
    gr.Markdown("# AI")
    state = gr.State()

    with gr.Tab("Chatbot"):
        with gr.Row():
            audio_recorder = gr.Audio(sources=["microphone"], label="Realtime Audio Streaming", streaming=True)

        markdown_display = gr.Markdown("Waiting...")

        audio_recorder.stream(
            fn=process_audio_streaming,
            inputs=[state, audio_recorder, markdown_display],
            outputs=[state, markdown_display]
        )

# demo.ssl_verify = False
demo.queue()
demo.launch(
    server_name="0.0.0.0",
    # share=True,
    max_threads=8,
    server_port=7861
)
