from snac import SNAC
from huggingface_hub import snapshot_download
from datasets import load_dataset
import torch
import torchaudio.transforms as T
from transformers import AutoTokenizer
import os
import soundfile as sf
import numpy as np
import librosa
from gradio_client import Client
import time
from tqdm import tqdm

from huggingface_hub import login
login(token="")

my_original_dataset_name = "MrDragonFox/Elise"
name_to_push_dataset_to = "BarryFutureman/kara"
dsn = my_original_dataset_name

ds = load_dataset(dsn, split="train", cache_dir="cache")
# # take first 100 for now
# ds = ds.select(range(100))
ds_sample_rate = ds[0]["audio"]["sampling_rate"]
print(f"Sample rate: {ds_sample_rate}")

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir="cache")
model = model.to("cuda")

RVC_CLIENT_URL = "http://localhost:7865/"
RVC_INDEX_PATH = "C:\\Files\\PythonProjects\\TTS\\Retrieval-based-Voice-Conversion-WebUI\\logs\\kara\\trained_IVF2477_Flat_nprobe_1_kara_v2.index"
RVC_OUTPUT_SR = 44000
RVC_EXPORT_FORMAT = "wav"
TARGET_SR = 24000

client = Client(RVC_CLIENT_URL)

def rvc_predict(input_path, output_path):
    result = client.predict(
        0,
        input_path,
        output_path,
        [input_path],
        2,
        "rmvpe",
        RVC_INDEX_PATH,
        RVC_INDEX_PATH,
        0.75,
        3,
        RVC_OUTPUT_SR,
        0.25,
        0.33,
        RVC_EXPORT_FORMAT,
        api_name="/infer_convert_batch"
    )
    output_file = os.path.join(output_path, os.path.basename(input_path)) + ".wav"
    return output_file

def rvc_batch_predict(input_paths, output_path):
    # Batch call to /infer_convert_batch
    client.predict(
        0,  # Speaker/Singer ID
        input_paths,  # List of input files
        output_path,  # Output directory
        input_paths,  # List of input files again (for batch)
        2,  # Transpose
        "rmvpe",  # Pitch extraction algorithm
        RVC_INDEX_PATH,  # Feature index file
        RVC_INDEX_PATH,  # Index path
        0.8,  # Search feature ratio
        3,  # Median filter radius
        RVC_OUTPUT_SR,  # Output sample rate
        0.25,  # Volume envelope scaling
        0.33,  # Protect voiceless consonants
        RVC_EXPORT_FORMAT,  # Export format
        api_name="/infer_convert_batch"
    )
    # Manually construct output file paths: {basename}.wav.wav
    output_files = [
        os.path.join(output_path, os.path.basename(p) + ".wav")
        for p in input_paths
    ]
    return output_files

def check_audio_length(orig_audio, converted_audio):
    orig_length = len(orig_audio)
    converted_length = len(converted_audio)
    if orig_length != converted_length:
        converted_audio = librosa.util.fix_length(converted_audio, size=orig_length)
    return orig_audio, converted_audio

def tokenize_audio(waveform):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to("cuda")

    # generate the codes from snac
    with torch.inference_mode():
        codes = model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
        all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

    return all_codes


def add_codes(example):
    # Always initialize codes_list to None
    codes_list = None

    try:
        answer_audio = example.get("audio")
        # If there's a valid audio array, tokenise it
        if answer_audio and "array" in answer_audio:
            audio_array = answer_audio["array"]
            codes_list = tokenize_audio(audio_array)
    except Exception as e:
        print(f"Skipping row due to error: {e}")
        # Keep codes_list as None if we fail
    example["codes_list"] = codes_list

    return example


tmp_audio_dir = "tmp_audio"
current_directory = os.getcwd()
rvc_output_dir = os.path.join(current_directory, "converted")
os.makedirs(tmp_audio_dir, exist_ok=True)
os.makedirs(rvc_output_dir, exist_ok=True)

batch_size = 16
audio_paths = []
orig_audios = []
sample_indices = []
for idx, sample in enumerate(ds):
    answer_audio = sample.get("audio")
    if answer_audio and "array" in answer_audio:
        orig_audio = answer_audio["array"]
        tmp_wav_path = os.path.join(tmp_audio_dir, f"sample_{idx}.wav")
        sf.write(tmp_wav_path, orig_audio, samplerate=ds_sample_rate)
        audio_paths.append(tmp_wav_path)
        orig_audios.append(orig_audio)
        sample_indices.append(idx)
    else:
        audio_paths.append(None)
        orig_audios.append(None)
        sample_indices.append(idx)

new_data = []
total = len(audio_paths)
processed = [False] * total
idx_map = {i: j for j, i in enumerate(sample_indices)}

# Add tqdm progress bar to the batch processing loop
for batch_start in tqdm(range(0, total, batch_size), desc="Converting audio batches"):
    batch_paths = [p for p in audio_paths[batch_start:batch_start+batch_size] if p is not None]
    batch_indices = [i for i in range(batch_start, min(batch_start+batch_size, total)) if audio_paths[i] is not None]
    if not batch_paths:
        continue
    rvc_wav_paths = rvc_batch_predict(batch_paths, rvc_output_dir)
    time.sleep(0.1)
    for k, i in enumerate(batch_indices):
        rvc_wav_path = rvc_wav_paths[k]
        orig_audio = orig_audios[i]
        if os.path.exists(rvc_wav_path):
            rvc_audio, rvc_sr = sf.read(rvc_wav_path)
            
            if rvc_sr != TARGET_SR:
                rvc_audio = librosa.resample(rvc_audio, orig_sr=rvc_sr, target_sr=TARGET_SR)
            _, rvc_audio = check_audio_length(orig_audio, rvc_audio)
            ds[i]["audio"] = rvc_audio.astype(np.float32)
        else:
            print(f"RVC output file not found: {rvc_wav_path}")
            ds[i]["audio"] = orig_audio
            raise ValueError(f"RVC output file not found: {rvc_wav_path}")
        processed[i] = True

ds.push_to_hub(name_to_push_dataset_to)
