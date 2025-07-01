import os
import time
import soundfile as sf
from datasets import Dataset, load_from_disk
from gradio_client import Client
import numpy as np
import librosa

TARGET_SR = 24000
RVC_CLIENT_URL = "http://localhost:7865/"
RVC_INDEX_PATH = "trained_IVF2477_Flat_nprobe_1_kara_v2.index"
RVC_OUTPUT_SR = 44000
RVC_EXPORT_FORMAT = "wav"

client = Client(RVC_CLIENT_URL)
print("Webui Client loaded.")


def rvc_predict(input_path, output_path):
    print("RVC Predicting...")
    result = client.predict(
        0,
        input_path,
        output_path,
        [input_path],
        0,
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
    print(result)
    output_file = os.path.join(output_path, os.path.basename(input_path)) + ".wav"
    print("output file", output_file)
    return output_file


def check_audio_length(orig_audio, converted_audio):
    orig_length = len(orig_audio)
    converted_length = len(converted_audio)

    print(f"Original audio length: {orig_length} samples")
    print(f"Converted audio length: {converted_length} samples")

    if orig_length != converted_length:
        print("Warning: The lengths of the original and converted audio are different!")
        print(orig_length, converted_length)
        # Stretch or shrink converted_audio to match orig_length
        converted_audio = librosa.util.fix_length(converted_audio, size=orig_length)
        print(f"Adjusted converted audio length: {len(converted_audio)} samples")

    return orig_audio, converted_audio

if __name__ == "__main__":
    dataset = load_from_disk("datasets/code2wav_hf_dataset_all")
    print(f"Loaded {len(dataset)} samples.")

    tmp_audio_dir = "tmp_audio"
    current_directory = os.getcwd()
    rvc_output_dir = current_directory + "/converted"
    os.makedirs(tmp_audio_dir, exist_ok=True)

    new_data = []
    for idx, sample in enumerate(dataset):
        orig_audio = sample["audio"]
        tmp_wav_path = os.path.join(tmp_audio_dir, f"sample_{idx}.wav")
        sf.write(tmp_wav_path, orig_audio, samplerate=TARGET_SR)
        rvc_wav_path = rvc_predict(tmp_wav_path, rvc_output_dir)
        time.sleep(0.02)
        if os.path.exists(rvc_wav_path):
            rvc_audio, rvc_sr = sf.read(rvc_wav_path)
            if rvc_sr != TARGET_SR:
                rvc_audio = librosa.resample(rvc_audio, orig_sr=rvc_sr, target_sr=TARGET_SR)
            # Check and adjust lengths if necessary
            orig_audio, rvc_audio = check_audio_length(orig_audio, rvc_audio)
            sample["original_audio"] = orig_audio
            sample["audio"] = rvc_audio.astype(np.float32)
        else:
            raise NotImplementedError(f"RVC output file not found: {rvc_wav_path}")
        new_data.append(sample)

    new_dataset = Dataset.from_list(new_data)
    new_dataset.save_to_disk("datasets/code2wav2rvc_hf_dataset")
