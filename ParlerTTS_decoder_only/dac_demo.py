import dac
from audiotools import AudioSignal
from parler_tts import DACModel


dac_model = DACModel.from_pretrained("parler-tts/parler_tts_mini_v0.1", cache_dir="cache/models")
dac_model.to('cuda')

# Load audio signal file
signal = AudioSignal("parler_tts_out.wav")

# Encode audio signal as one long file
# (may run out of GPU memory on long files)
signal.to(dac_model.device)
output = dac_model.encode(signal.audio_data, sample_rate=signal.sample_rate)

print(output.audio_codes.shape)

quit()

# Decode audio signal
y = model.decode(z)

# Alternatively, use the `compress` and `decompress` functions
# to compress long files.

signal = signal.cpu()
x = model.compress(signal)

# Save and load to and from disk
x.save("compressed.dac")
x = dac.DACFile.load("compressed.dac")

# Decompress it back to an AudioSignal
y = model.decompress(x)

# Write to file
y.write('output.wav')
