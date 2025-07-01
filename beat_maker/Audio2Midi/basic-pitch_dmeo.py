from basic_pitch.inference import predict_and_save
import os

input_file_name = "once-in-paris-168895.mp3"
# https://signal.vercel.app/edit
# https://samplette.io
# https://www.youtube.com/watch?v=v47lmqfrQ9s
output_dir = "output/" + input_file_name.replace(".mp3", " MIDI")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

predict_and_save(
    audio_path_list=[input_file_name],
    output_directory=output_dir,
    save_midi=True,
    sonify_midi=False,
    save_model_outputs=False,
    save_notes=False,
    onset_threshold=0.5,  # Lowers gives more small chunks
    frame_threshold=0.3,
    minimum_note_length=58*2,
    minimum_frequency=None,
    maximum_frequency=None,
    multiple_pitch_bends=False,
    melodia_trick=False,
    debug_file=None,
    sonification_samplerate=44100,
    midi_tempo=120,
)