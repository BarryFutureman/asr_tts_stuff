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
import difflib
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch.nn.functional as F


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


class TokenCandidate:
    def __init__(self, token_id: int, token_text: str, probability: float):
        self.token_id = token_id
        self.token_text = token_text
        self.probability = probability

    def update_probability(self, new_prob: float):
        """Update probability using exponential moving average"""
        self.probability = self.probability * 0.9 + new_prob

    def __repr__(self):
        return f"{self.token_text}({self.probability:.2f})"


class TranscribedChunk:
    def __init__(self, token_candidates_list: list = None, is_fixed: bool = False):
        # List of lists - each inner list contains 5 TokenCandidate objects for that position
        self.token_candidates_list = token_candidates_list or []
        self.is_fixed = is_fixed

    def get_high_confidence_text(self, confidence_threshold: float = 2.0):
        """Get text using only tokens with probability > threshold"""
        text_parts = []
        for position_candidates in self.token_candidates_list:
            # Get the highest probability candidate
            best_candidate = max(position_candidates, key=lambda x: x.probability)
            if best_candidate.probability > confidence_threshold:
                text_parts.append(best_candidate.token_text)
            else:
                break  # Stop at first low-confidence token
        return "".join(text_parts)

    def get_full_text(self):
        """Get text using the most probable token at each position"""
        text_parts = []
        for position_candidates in self.token_candidates_list:
            best_candidate = max(position_candidates, key=lambda x: x.probability)
            text_parts.append(best_candidate.token_text)
        return "".join(text_parts)

    def update_probabilities(self, new_token_candidates_list):
        """Update probabilities for existing tokens and add new ones"""
        # Ensure we have the same length
        min_len = min(len(self.token_candidates_list), len(new_token_candidates_list))
        
        # Update existing positions
        for i in range(min_len):
            existing_candidates = {tc.token_id: tc for tc in self.token_candidates_list[i]}
            
            for new_candidate in new_token_candidates_list[i]:
                if new_candidate.token_id in existing_candidates:
                    existing_candidates[new_candidate.token_id].update_probability(new_candidate.probability)
                else:
                    # Add new candidate with initial probability
                    self.token_candidates_list[i].append(new_candidate)
            
            # Keep only top 5 candidates
            self.token_candidates_list[i].sort(key=lambda x: x.probability, reverse=True)
            self.token_candidates_list[i] = self.token_candidates_list[i][:5]
        
        # Add new positions
        if len(new_token_candidates_list) > len(self.token_candidates_list):
            self.token_candidates_list.extend(new_token_candidates_list[min_len:])

    def __repr__(self):
        text = self.get_full_text()
        return f"{text}[{'fixed' if self.is_fixed else 'active'}]"


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
        model_id = "openai/whisper-large-v3-turbo"

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

    def get_partial_text_context(self, confidence_threshold=2.0, remove_last_n_tokens=3):
        if not self.full_running_transcription:
            return "", "", ""

        # Get high-confidence text from all chunks except the last one
        full_text_without_last_chunk = "".join(
            chunk.get_high_confidence_text(confidence_threshold) 
            for chunk in self.full_running_transcription[:-1]
        )

        if not self.full_running_transcription or self.full_running_transcription[-1].is_fixed:
            return full_text_without_last_chunk, "", ""

        last_chunk = self.full_running_transcription[-1]
        
        # Get high-confidence tokens from last chunk
        high_conf_tokens = []
        removed_tokens = []
        
        for i, position_candidates in enumerate(last_chunk.token_candidates_list):
            best_candidate = max(position_candidates, key=lambda x: x.probability)
            if i < len(last_chunk.token_candidates_list) - remove_last_n_tokens:
                if best_candidate.probability > confidence_threshold:
                    high_conf_tokens.append(best_candidate.token_text)
            else:
                removed_tokens.append(best_candidate.token_text)

        partial_text = "".join(high_conf_tokens)
        removed_words = "".join(removed_tokens)
        full_partial_text = full_text_without_last_chunk + partial_text

        return full_partial_text, partial_text, removed_words

    def get_full_text(self):
        return "".join(chunk.get_full_text() for chunk in self.full_running_transcription)

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
    
    def post_process_transcription(self):
        # Joint all fixed chunks into a single string
        fixed_chunks = [chunk for chunk in self.full_running_transcription if chunk.is_fixed]
        joined_text = "".join(chunk.get_full_text() for chunk in fixed_chunks)
        self.full_running_transcription = [TranscribedChunk(joined_text, is_fixed=True)] + [chunk for chunk in self.full_running_transcription if not chunk.is_fixed]

    def detect_repetition(self, new_text, lookback_chunks=8):
        """
        Detect if the new_text is a repetition of recent transcription chunks.
        Returns True if repetition is detected.
        """
        if not self.full_running_transcription or not new_text.strip():
            return False
        for lookback in range(2, lookback_chunks):
            # Get recent text from the last few chunks for comparison
            recent_chunks = self.full_running_transcription[-lookback:]
            recent_text = "".join(chunk.get_full_text() for chunk in recent_chunks).strip()

            if not recent_text:
                return False

            new_text_clean = new_text.strip()

            # Check for exact substring match
            if new_text_clean in recent_text:
                return True

            # Check for high similarity using difflib
            def fuzzy_contains(long_string, short_string, threshold=0.8):
                """
                Check if short_string fuzzily appears in long_string using SequenceMatcher.
                Returns True if any substring of long_string has a similarity ratio above threshold.
                """
                len_short = len(short_string)
                for i in range(len(long_string) - len_short + 1):
                    window = long_string[i:i + len_short]
                    ratio = difflib.SequenceMatcher(None, window, short_string).ratio()
                    if ratio >= threshold:
                        return True
                return False

            if fuzzy_contains(recent_text, new_text_clean):
                return True
        
        return False

    def transcribe_audio_with_probabilities(self, base64_audio):
        """Transcribe audio and return top 5 token candidates for each position"""
        # Convert base64 audio back to numpy array
        sr, audio_data = base64_to_numpy_audio(base64_audio)

        target_sr = 16000
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)

        audio_data = audio_data[-target_sr * 16:]

        audio_data = self.vad_step(audio_data)
        if audio_data is None:
            return []

        # Get partial text context from existing transcription
        full_partial_text, part_partial_text, removed_words = self.get_partial_text_context(
            confidence_threshold=2.0, remove_last_n_tokens=8
        )

        # Prepare inputs for Whisper
        inputs = self.processor(audio_data, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device).to(self.torch_dtype) if v.dtype == torch.float32 else v.to(self.device)
                  for k, v in inputs.items()}

        # Create decoder input with context
        decoder_start_token_id = self.model.config.decoder_start_token_id
        lang_token = self.processor.tokenizer.convert_tokens_to_ids("<|en|>")
        task_token = self.processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_token = self.processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

        prompt_tokens = self.processor.tokenizer.encode(full_partial_text, add_special_tokens=False) if full_partial_text else []
        decoder_input_ids = [decoder_start_token_id, lang_token, task_token, notimestamps_token] + prompt_tokens
        decoder_input_ids = torch.tensor([decoder_input_ids], dtype=torch.long).to(self.device)

        # Generate with probability tracking
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_features"],
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=16,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )

        # Extract top 5 tokens for each position
        token_candidates_list = []
        for score in outputs.scores:  # scores for each generated token
            # Get probabilities
            probs = F.softmax(score[0], dim=-1)
            
            # Get top 5 tokens
            top_probs, top_indices = torch.topk(probs, 5)
            
            position_candidates = []
            for prob, token_id in zip(top_probs, top_indices):
                token_text = self.processor.tokenizer.decode([token_id], skip_special_tokens=True)
                position_candidates.append(TokenCandidate(
                    token_id=token_id.item(),
                    token_text=token_text,
                    probability=prob.item()
                ))
            
            token_candidates_list.append(position_candidates)

        return token_candidates_list

    def transcribe_audio(self, base64_audio):
        token_candidates_list = self.transcribe_audio_with_probabilities(base64_audio)
        
        if not token_candidates_list:
            return "..."

        # Get context for logging
        full_partial_text, part_partial_text, removed_words = self.get_partial_text_context()
        
        if not self.full_running_transcription:
            # First chunk
            self.full_running_transcription.append(
                TranscribedChunk(token_candidates_list, is_fixed=False)
            )
        else:
            last_chunk = self.full_running_transcription[-1]
            
            # Check if we should update existing chunk or create new one
            if not last_chunk.is_fixed and len(last_chunk.token_candidates_list) > 0:
                # Update probabilities in existing chunk
                last_chunk.update_probabilities(token_candidates_list)
                
                # Check if we should fix some tokens (high confidence tokens)
                high_conf_positions = 0
                for position_candidates in last_chunk.token_candidates_list:
                    best_candidate = max(position_candidates, key=lambda x: x.probability)
                    if best_candidate.probability > 3.0:  # Very high confidence threshold for fixing
                        high_conf_positions += 1
                    else:
                        break
                
                # If we have many high-confidence tokens, split the chunk
                if high_conf_positions > 8:
                    # Split chunk - first part becomes fixed
                    fixed_candidates = last_chunk.token_candidates_list[:high_conf_positions-4]
                    remaining_candidates = last_chunk.token_candidates_list[high_conf_positions-4:]
                    
                    last_chunk.token_candidates_list = fixed_candidates
                    last_chunk.is_fixed = True
                    
                    # Add new active chunk
                    self.full_running_transcription.append(
                        TranscribedChunk(remaining_candidates, is_fixed=False)
                    )
            else:
                # Create new chunk
                self.full_running_transcription.append(
                    TranscribedChunk(token_candidates_list, is_fixed=False)
                )

        # Clean up old chunks
        if len(self.full_running_transcription) > 8:
            # Merge first two chunks
            first_chunk = self.full_running_transcription[0]
            second_chunk = self.full_running_transcription[1]
            
            merged_candidates = first_chunk.token_candidates_list + second_chunk.token_candidates_list
            merged_chunk = TranscribedChunk(merged_candidates, is_fixed=True)
            
            self.full_running_transcription = [merged_chunk] + self.full_running_transcription[2:]

        # Generate debug info
        generated_text = "".join(
            max(pos_candidates, key=lambda x: x.probability).token_text 
            for pos_candidates in token_candidates_list
        )
        
        # Create probability summary
        prob_summary = []
        for i, pos_candidates in enumerate(token_candidates_list[:5]):  # Show first 5 positions
            best = max(pos_candidates, key=lambda x: x.probability)
            prob_summary.append(f"{best.token_text}({best.probability:.2f})")
        
        print(f"Partial context: '{full_partial_text}'")
        print(f"Generated: '{generated_text}'")
        print(f"Token probs: {' '.join(prob_summary)}")
        print(f"Chunks: {''.join([str(chunk) for chunk in self.full_running_transcription])}")

        return (
            f"## {self.get_full_text()}  \n"
            f"**Partial context:** `{full_partial_text}`  \n"
            f"**Generated:** `{generated_text}`  \n"
            f"**Token probs:** `{' '.join(prob_summary)}`  \n"
            f"**Chunks:** `{''.join([str(chunk) for chunk in self.full_running_transcription])}`"
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
