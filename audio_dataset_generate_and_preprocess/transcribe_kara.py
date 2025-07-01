import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
from datasets import Dataset, Features, Audio, Value

import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Directory containing audio files
wav_dir = "Detroit Become Human Kara Sounds"
name_to_push_dataset_to = "BarryFutureman/kara2_filtered"

# HuggingFace Whisper model setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "nyrahealth/CrisperWhisper"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, cache_dir="cache"
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=False,
    torch_dtype=torch_dtype,
    device=device,
)

# Load Silero VAD model and utils
silero_model, silero_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
silero_model.to("cpu")
(get_speech_timestamps, _, read_audio, _, _) = silero_utils

batch_size = 1
data = []

file_list = [fname for fname in os.listdir(wav_dir) if fname.lower().endswith(".mp3")]

for idx in tqdm(range(0, len(file_list), batch_size), desc="Transcribing"):
    batch_files = file_list[idx:idx+batch_size]
    audio_paths = [os.path.abspath(os.path.join(wav_dir, fname)) for fname in batch_files]
    audio_inputs = []
    trimmed_audios = []  # <-- Add this line
    for path in audio_paths:
        audio, sr = sf.read(path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        vad_sr = 16000
        fade_sec = 1.0
        # Resample for VAD
        if sr != vad_sr:
            audio_vad = librosa.resample(audio, orig_sr=sr, target_sr=vad_sr)
        else:
            audio_vad = audio
        audio_tensor = torch.tensor(audio_vad, dtype=torch.float32, device="cpu")
        speech_timestamps = get_speech_timestamps(audio_tensor, silero_model, return_seconds=False)
        # Default to full audio if no speech detected
        if speech_timestamps:
            start = speech_timestamps[0]['start']
            end = speech_timestamps[-1]['end']
        else:
            start = 0
            end = len(audio_vad)
        # Apply 0.2s fade in and 1s fade out
        fade_in_sec = 0.2
        fade_out_sec = 1.0
        start = max(0, start - int(fade_in_sec * vad_sr))
        end = min(len(audio_vad), end + int(fade_out_sec * vad_sr))
        # Convert indices to original sample rate
        if sr != vad_sr:
            start = int(start * sr / vad_sr)
            end = int(end * sr / vad_sr)

        # Pad with silence at the beginning and end if needed
        pad_left = max(0, int(fade_in_sec * sr) - start)
        pad_right = max(0, (end + int(fade_out_sec * sr)) - len(audio))
        trimmed = audio[max(0, start):min(len(audio), end)]

        if pad_left > 0:
            trimmed = np.concatenate([np.zeros(pad_left, dtype=trimmed.dtype), trimmed])
        if pad_right > 0:
            trimmed = np.concatenate([trimmed, np.zeros(pad_right, dtype=trimmed.dtype)])

        # For debugging: save trimmed audio
        trimmed_path = os.path.join("./tmp_audio", f"trimmed_{os.path.basename(path)}")
        sf.write(trimmed_path, trimmed, sr)
        
        audio_inputs.append({"array": trimmed, "sampling_rate": sr})
        trimmed_audios.append(trimmed)

    results = pipe(audio_inputs)
    for fname_out, result, audio_array in zip(batch_files, results, trimmed_audios):
        result_text = result["text"]
        # Filter: skip if empty or contains "thank you"
        if not result_text.strip():
            continue
        if "thank you" in result_text.lower():
            continue
        # Replace [UH] with ...
        result_text = result_text.replace("[UH]", "...")
        result_text = result_text.replace("[UM]", "um")

        # --- Enhance text with tag from filename ---
        base = os.path.splitext(os.path.basename(fname_out))[0]
        parts = base.split('_')
        # Remove all parts that start with 'X'
        parts = [p for p in parts if not p.startswith('X')]
        # Remove "ENG", "PC", "FA"
        parts = [p for p in parts if p not in ("ENG", "PC", "FA")]
        tag = "_".join(parts).lower()
        tag = f"<{tag}>"
        result_text = f"{tag} {result_text}"
        # --- End enhancement ---
        # Save trimmed audio array and sampling rate in the dataset
        audio_array = np.array(audio_array, dtype=np.float32)
        audio_dict = {"array": audio_array, "sampling_rate": sr}
        data.append({
            "audio": audio_dict,
            "text": result_text
        })
        print(f"{fname_out}: {result_text}")

# Create HuggingFace dataset with Audio feature
features = Features({
    "audio": Audio(sampling_rate=None),
    "text": Value("string"),
})
hf_dataset = Dataset.from_list(data, features=features)
print(hf_dataset.features)
# hf_dataset.save_to_disk(os.path.join(wav_dir, "kara_transcriptions_dataset"))
hf_dataset.push_to_hub(name_to_push_dataset_to)
