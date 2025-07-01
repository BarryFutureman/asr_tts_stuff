import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_path = 'songs/NEONI - VILLAIN (Lyrics).mp3'
y, sr = librosa.load(audio_path)

# Use onset detection to find beats
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# Convert beat frames to time
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# Plot waveform and beats
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(y)) / sr, y, alpha=0.6)
plt.vlines(beat_times, -1, 1, color='r', linestyle='--', alpha=0.8, label='Beats')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform with Detected Beats')
plt.legend()
plt.show()

print(f'Tempo: {tempo} BPM')
print(f'Number of beats detected: {len(beat_times)}')
