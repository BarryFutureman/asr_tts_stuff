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
from PIL import Image
import threading
import queue

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa


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

        model_id = "firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53"
        self.model = AutoModelForAudioClassification.from_pretrained(model_id).to("cpu")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True,
                                                                      return_attention_mask=True)
        self.id2label = self.model.config.id2label
        
        self.result_queue.put("Sample")
        self.speaker_audio = None
        self.load_speaker_audio()

    def preprocess_audio(self, base64_audio, feature_extractor, max_duration=30.0):
        # Decode base64 string to bytes
        audio_bytes = base64.b64decode(base64_audio)

        # Load audio using librosa from a BytesIO stream
        audio_buffer = io.BytesIO(audio_bytes)
        audio_array, sampling_rate = librosa.load(audio_buffer, sr=feature_extractor.sampling_rate)

        # Pad or truncate to max_length
        max_length = int(feature_extractor.sampling_rate * max_duration)
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        else:
            audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

        # Extract features
        inputs = feature_extractor(
            audio_array,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return inputs

    def predict_emotion(self, base64_audio, model, feature_extractor, id2label, max_duration=30.0):
        inputs = self.preprocess_audio(base64_audio, feature_extractor, max_duration)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_label = id2label[predicted_id]

        return predicted_label

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
            # Clear all items in the queue except the last one
            last_task = (sr, data, text_prompt)
            while not self.task_queue.empty():
                try:
                    last_task = self.task_queue.get_nowait()
                except queue.Empty:
                    break
            sr, data, text_prompt = last_task
            
            
            base64_audio = numpy_to_base64_wav(data, sr)

            predicted_label = self.predict_emotion(base64_audio, self.model, self.feature_extractor, self.id2label)
            print(predicted_label)
            # Save audio with label in filename
            import os
            os.makedirs("./tmp", exist_ok=True)
            filename = f"./tmp/audio_{predicted_label}_{int(time.time())}.wav"
            sf.write(filename, data, sr)
            self.result_queue.put(predicted_label)



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
        audio_analyser.task_queue.put((sr, audio[-sr * 8:], text_prompt))
        
        # Determine the emotion from the result
        emotion = "neutral"
        for em in ALL_EMOTIONS:
            if em in result.lower():
                emotion = em
                break
        
        # Load the corresponding emotion image
        face_display = f"emotions/{emotion}.png"
        
        return audio, f"Task in after: {result}", Image.open(face_display)
    except queue.Empty:
        result = previous_result

    return audio, result, face_display


with gr.Blocks(title="My Recorder", theme=Softy()) as demo:
    gr.Markdown("## Recorder")
    state = gr.State()

    with gr.Tab("Audio Recorder"):
        speaker_audio_recorder = gr.Audio(sources="microphone", label="Speaker Audio", streaming=False)
        submit_button = gr.Button("Submit Speaker Audio")
        audio_recorder = gr.Audio(sources="microphone", label="Realtime Audio Streaming", streaming=True)

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
    # ssl_keyfile="key.pem",
    # ssl_certfile="cert.pem",
    ssl_verify=False
)
