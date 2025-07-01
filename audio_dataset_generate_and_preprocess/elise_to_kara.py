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


from huggingface_hub import login
login(token="")

my_original_dataset_name = "MrDragonFox/Elise"
name_to_push_dataset_to = "BarryFutureman/kara-tokenised-snac"
dsn = my_original_dataset_name

ds = load_dataset(dsn, split="train", cache_dir="cache")
# take first 100 for now
ds = ds.select(range(100))
ds_sample_rate = ds[0]["audio"]["sampling_rate"]
print(f"Sample rate: {ds_sample_rate}")

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir="cache")
model = model.to("cuda")

RVC_CLIENT_URL = "http://localhost:7865/"
RVC_INDEX_PATH = "trained_IVF2477_Flat_nprobe_1_kara_v2.index"
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
        0,  # Transpose
        "rmvpe",  # Pitch extraction algorithm
        RVC_INDEX_PATH,  # Feature index file
        RVC_INDEX_PATH,  # Index path
        0.75,  # Search feature ratio
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

for batch_start in range(0, total, batch_size):
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


ds = ds.map(add_codes)  # , remove_columns=["audio"])

tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai =  tokeniser_length + 6
pad_token = tokeniser_length + 7

audio_tokens_start = tokeniser_length + 10

tokenizer_name = "canopylabs/orpheus-3b-0.1-pretrained"


tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir="cache")
num_proc = os.cpu_count() - 2

ds = ds.filter(lambda x: x["codes_list"] is not None)
ds = ds.filter(lambda x: len(x["codes_list"]) > 0)


def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = vals[:7]

    removed_frames = 0

    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(vals[i:i+7])
        else:
            removed_frames += 1

    example["codes_list"] = result

    return example


ds = ds.map(remove_duplicate_frames, num_proc=num_proc)

tok_info = '''*** HERE you can modify the text prompt
i.e. if you wanted a multispeaker model like canopylabs/orpheus-3b-0.1-ft, you can pass:
f"{example["source"]}:  {example["text"]}", as is passed.
'''
print(tok_info)


def create_input_ids(example):
    text_ids = tokenizer.encode(example["text"],  add_special_tokens=True)
    text_ids.append(end_of_text)
    example["text_tokens"] = text_ids
    input_ids = (
        [start_of_human]
        + example["text_tokens"]
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
    )
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)

    return example


ds = ds.map(create_input_ids, num_proc=num_proc, remove_columns=["text", "codes_list"])


columns_to_keep = ["input_ids", "labels", "attention_mask", "audio"]
columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]

ds = ds.remove_columns(columns_to_remove)

ds.push_to_hub(name_to_push_dataset_to)
