import time
from typing import Annotated, AsyncGenerator, Literal

import gradio as gr
import numpy as np
import torch
from gradio.themes.soft import Soft
from gradio.themes.utils import colors, fonts, sizes
import json
import requests

import soundfile as sf
import base64
import io
from dataclasses import dataclass, field
from typing import *
import threading
import queue


ALL_EMOTIONS = ['disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', "contempt", "angry"]


class Softy(Soft):
    def __init__(
            self,
            *,
            primary_hue: colors.Color | str = colors.rose,
            secondary_hue: colors.Color | str = colors.gray,
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=colors.stone,
        )
        super().set(

        )


# Save NumPy array to WAV
def numpy_to_base64_wav(data, sr):
    with io.BytesIO() as wav_io:
        sf.write(wav_io, data, sr, format="wav")
        wav_io.seek(0)
        base64_audio = base64.b64encode(wav_io.read()).decode("utf-8")
    return base64_audio


def submit_speaker_audio(audio):
    sr, data = audio
    sf.write("speaker.wav", data, sr)
    with io.BytesIO() as wav_io:
        sf.write(wav_io, data, sr, format="wav")
        wav_io.seek(0)
        base64_audio = base64.b64encode(wav_io.read()).decode("utf-8")

        audio_analyser.speaker_audio = base64_audio


class AudioAnalyser:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        
        self.result_queue.put("Sample")
        self.speaker_audio = None
        self.load_speaker_audio()

    def load_speaker_audio(self):
        try:
            with open("speaker.wav", "rb") as wav_file:
                base64_audio = base64.b64encode(wav_file.read()).decode("utf-8")
                self.speaker_audio = base64_audio
        except FileNotFoundError:
            print("Speaker audio file not found. Please submit speaker audio.")

    def run(self):
        while True:
            sr, data, text_prompt = self.task_queue.get()
            base64_audio = numpy_to_base64_wav(data, sr)
            conversation = [
                {"role": "system", "content": [
                    {"type": "audio", "audio_url": "data:audio/wav;base64," + self.speaker_audio},
                    {"type": "text", "text": f"This is the speaker's voice, remember it. In the following audio we will analyse the speaker's emotion in a conversation.\nemotions = {str(ALL_EMOTIONS)}. The speaker should have neutral emotion most of the time."},
                ]},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "data:audio/wav;base64," + base64_audio},
                    {"type": "text", "text": text_prompt},
                ]}
            ]
            response = requests.post("http://dh2010pc01.utm.utoronto.ca:5069/process_audio", json=conversation)
            self.result_queue.put(response.json()["response"])


audio_analyser = AudioAnalyser()


def process_audio_streaming(audio, new_chunk, text_prompt, previous_result, face_display):
    sr, y = new_chunk
    if audio is not None:
        audio = np.concatenate([audio, y])
    else:
        audio = y

    sf.write("audio.wav", audio, sr)

    try:
        result = audio_analyser.result_queue.get_nowait()
        audio_analyser.task_queue.put((sr, audio[-sr * 20:], text_prompt))
        
        # Determine the emotion from the result
        emotion = "neutral"
        for em in ALL_EMOTIONS:
            if em in result.lower():
                emotion = em
                break
        
        # Load the corresponding emotion image
        face_display = f"emotions/{emotion}.png"
        
        return audio, f"Task in after: {result}", face_display
    except queue.Empty:
        result = previous_result

    return audio, result, face_display


with gr.Blocks(title="My Recorder", theme=Softy()) as demo:
    gr.Markdown("## Recorder")
    state = gr.State()

    with gr.Tab("Audio Recorder"):
        speaker_audio_recorder = gr.Audio(sources=["microphone"], label="Speaker Audio", streaming=False)
        submit_button = gr.Button("Submit Speaker Audio")
        audio_recorder = gr.Audio(sources=["microphone"], label="Realtime Audio Streaming", streaming=True,
                                  waveform_options=gr.WaveformOptions(waveform_color="#B83A4B"))

        emotion_display = gr.Image()

        text_input = gr.Textbox(label="Enter your text prompt here", value="Carefully think through what they are saying. Produce a transcript. Identify the speaker, if the target speaker is not in the audio just say None.\nSpeaker's current emotion by the end of the audio:")

        text_output_markdown = gr.Markdown(value="# OUTPUT")

        audio_recorder.stream(fn=process_audio_streaming,
                              inputs=[state, audio_recorder, text_input, text_output_markdown, emotion_display],
                              outputs=[state, text_output_markdown, emotion_display])

        submit_button.click(
            fn=submit_speaker_audio,
            inputs=[speaker_audio_recorder],
            outputs=None
        )

# demo.ssl_verify = False
demo.queue()
demo.launch(
    server_name="0.0.0.0",
    # share=True,
    max_threads=8,
    server_port=7860,
    ssl_keyfile="key.pem",
    ssl_certfile="cert.pem",
    ssl_verify=False
)
