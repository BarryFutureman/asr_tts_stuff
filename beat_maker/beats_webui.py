import time

import gradio as gr
from gradio.themes.soft import Soft
from gradio.themes.utils import colors, fonts, sizes
import youtube_audio_download
import librosa
from io import BytesIO
import numpy as np


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


def youtube_to_audio(link_url):
    return youtube_audio_download.download_audio_video(link_url)


def generate_beats(audio_data):
    print(audio_data)
    sample_rate, audio_array = audio_data[0], audio_data[1]
    audio_array = audio_array[:, 0]  # single channel

    audio_array = audio_array.astype(np.float32) / 32768.0
    print(audio_array)

    # Detect beats
    tempo, beat_frames = librosa.beat.beat_track(y=audio_array, sr=sample_rate)

    # Convert beat timings to seconds
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)

    beat_intervals = np.diff(beat_times)
    average_interval = np.mean(beat_intervals)

    # Define a threshold to differentiate between big and small beats
    interval_threshold = 1 * average_interval  # You can adjust this threshold based on your music and preference

    # Calculate average amplitude for each beat
    beat_amplitudes = []
    for i in range(len(beat_frames) - 1):
        start = beat_frames[i]
        end = beat_frames[i + 1]
        beat_amplitude = np.mean(np.abs(audio_array[start:end]))
        beat_amplitudes.append(beat_amplitude)

    amplitude_threshold = np.mean(beat_amplitudes) / 2

    # Separate big beats and small beats based on the threshold
    beats_lst = []
    for i in range(len(beat_intervals)):
        if beat_intervals[i] > interval_threshold:
            info_tuple = beat_times[i], "big_interval"
            beats_lst.append(info_tuple)
        else:
            info_tuple = beat_times[i], "small_interval"
            beats_lst.append(info_tuple)
        if beat_amplitudes[i] > amplitude_threshold:
            info_tuple = beat_times[i], "heavy"
            beats_lst.append(info_tuple)

    beat_string = ""
    # Write beat timestamps to a text file
    with open("beats.txt", 'w') as beat_f:
        for info_tuple in beats_lst:
            beat_time, beat_type = info_tuple
            beat_f.write(f'{beat_time:.2f} - {beat_type}\n')
            beat_string += "#### " + str(beat_time) + " "

    return gr.Markdown(beat_string)


def play_beats():
    play_start_time = time.time()

    beat_string = ""
    with open("beats.txt", "r") as beat_f:
        line = beat_f.readline()
        while line:
            t, b = line.strip().split(" - ")
            beat_time = float(t)
            if play_start_time + beat_time < time.time():
                line = beat_f.readline()
                if b == "heavy":
                    beat_string += "\n### " + str(beat_time) + b + " "
                else:
                    beat_string += "#### " + str(beat_time) + b + " "
                yield gr.Markdown(beat_string)


with gr.Blocks(title="Beats", theme=Softy()) as demo:
    gr.Markdown("## BEATS")
    # Guess tab
    with gr.Tab("Beats") as audio_craft_tab:
        with gr.Column(variant="panel"):
            audio_input = gr.Audio()
            gen_btn = gr.Button(value="Generate Beats", variant="primary")
            output_md = gr.Markdown()

            gen_btn.click(fn=generate_beats, inputs=[audio_input], outputs=output_md)
            audio_input.play(fn=play_beats, inputs=[], outputs=output_md)

    with gr.Tab("Youtube") as youtube_tab:
        youtube_vid_url_textbox = gr.Textbox(label="YouTube video link")
        audio_input = gr.Audio()
        gen_btn = gr.Button(value="Retrieve Audio", variant="primary")

        gen_btn.click(fn=youtube_to_audio, inputs=youtube_vid_url_textbox, outputs=audio_input)

        with open("youtube_audio_download.py", "r") as f:
            gr.Code(language="python", value=f.read())
            f.close()

demo.queue()
demo.launch(max_threads=8,  server_port=7861, share=False)
