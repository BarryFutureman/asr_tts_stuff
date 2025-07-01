import time

import gradio as gr
from gradio.themes.soft import Soft
from gradio.themes.utils import colors, fonts, sizes
import youtube_audio_download
import librosa
from io import BytesIO
import numpy as np


dark_color = colors.Color(
    name="dark_color",
    c50="#eceff1",
    c100="#e0e0e0",
    c200="#bdbdbd",
    c300="#9e9e9e",
    c400="#757575",
    c500="#616161",
    c600="#424242",
    c700="#212121",
    c800="#000000",
    c900="#000000",
    c950="#000000",
)


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
            neutral_hue=dark_color,
            font=fonts.GoogleFont("Sora")
        )
        super().set(

        )


def play_text(text_to_play):
    time.sleep(1)
    prefix = "\n" * 6 + "## "

    text_to_display = ""
    for char in text_to_play:
        if char == "\n":
            time.sleep(1)
        text_to_display += char
        time.sleep(0.02)
        yield gr.Markdown(prefix+text_to_display)


with gr.Blocks(title="Anim Text", theme=Softy()) as demo:
    # Guess tab
    with gr.Tab("Beats") as audio_craft_tab:
        # with gr.Column(variant="panel"):
        gr.Markdown("### \n" * 16)
        with gr.Row():
            text_display_md = gr.Markdown(scale=2)
            with gr.Column(variant="panel", scale=1):
                text_input = gr.Textbox(interactive=True)
                play_btn = gr.Button(value="Animate", variant="primary")

                play_btn.click(fn=play_text, inputs=[text_input], outputs=[text_display_md])

demo.queue()
demo.launch(max_threads=8,  server_port=7861, share=False)
