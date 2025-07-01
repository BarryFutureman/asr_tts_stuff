import librosa
import numpy as np

def detect_beats(audio_file, output_file):
    # Load the audio file
    y, sr = librosa.load(audio_file)
    print(y)
    quit()

    # Detect beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # Convert beat timings to seconds
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Write beat timestamps to a text file
    with open(output_file, 'w') as f:
        for beat_time in beat_times:
            f.write(f'{beat_time:.2f}\n')

    print(f"Beat timestamps written to {output_file}")

if __name__ == "__main__":
    audio_file = "songs/NEONI - VILLAIN (Lyrics).mp3"  # Replace with the path to your audio file
    output_file = "beat_timestamps.txt"  # Output file to store beat timestamps
    detect_beats(audio_file, output_file)
