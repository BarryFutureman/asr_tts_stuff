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


class RealtimeVoiceProcessor:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        self.result_queue.put(ProcessorResult(0, "INIT"))
        
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
        self.full_running_transcription = ""
        
    def deduplicate_text(self, existing_text, new_text):
        if not existing_text.strip():
            return new_text.strip()

        if not new_text.strip():
            return ""
            
        try:
            # Prepare request data for LLM server
            payload = {
                "past_context": existing_text,
                "new_transcription": new_text,
                "max_tokens": 200,
                "temperature": 0.3
            }
            
            # Make request to LLM deduplication server
            response = requests.post(
                "http://127.0.0.1:8069/deduplication",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                updated_context = result.get("updated_context", "")
                
                return existing_text + updated_context
            else:
                print(f"LLM server error: {response.status_code}")
                raise NotImplementedError()
                
        except Exception as e:
            print(f"Error calling LLM server: {e}")
            # Fallback to simple deduplication
            return existing_text
        
    def transcribe_audio(self, base64_audio):
        # Convert base64 audio back to numpy array
        sr, audio_data = base64_to_numpy_audio(base64_audio)

        target_sr = 16000
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)

        audio_data = audio_data[-target_sr * 4:]
        
        # Transcribe using the model
        output = self.model.transcribe([audio_data], timestamps=False)
        
        transcribed_text = output[0].text

        # Apply deduplication to prevent repeated text
        deduplicated_text = self.deduplicate_text(self.full_running_transcription, transcribed_text)
        
        self.full_running_transcription += deduplicated_text

        print(output)
        
        return transcribed_text

    def run(self):
        while True:
            sr, data = self.task_queue.get()
            input_audio_buffer_size = len(data)

            # if not np.issubdtype(data.dtype, np.floating):
            #     data = data.astype(np.float32)
            #     data /= 32768.0

            base64_audio = numpy_to_base64_wav(data, sr)
            transcribed_text = self.transcribe_audio(base64_audio=base64_audio)

            self.result_queue.put(ProcessorResult(input_audio_buffer_size=0, text=self.full_running_transcription))


def process_audio_streaming(audio_buffer, new_chunk, previous_text_display):
    sr, y = new_chunk
    if audio_buffer is not None:
        audio_buffer = np.concatenate([audio_buffer, y])
    else:
        audio_buffer = y

    try:
        if len(audio_buffer) < sr * 4:
            raise queue.Empty

        processed_result = realtime_audio_processor.result_queue.get_nowait()

        # Trim audio_buffer to only include new audio since last turn
        input_audio_buffer_size = processed_result.input_audio_buffer_size
        # The audio_buffer passed to task_queue should be the audio for *this* turn
        current_turn_audio_data = audio_buffer[input_audio_buffer_size:]
        # Update audio_buffer to remove processed part for next iteration's state
        audio_buffer = audio_buffer[input_audio_buffer_size:]

        realtime_audio_processor.task_queue.put((sr, current_turn_audio_data))

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
