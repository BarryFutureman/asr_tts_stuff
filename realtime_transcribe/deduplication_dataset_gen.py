import os
import random
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import soundfile as sf
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
from tqdm import tqdm

# Global variable to control number of items to process
NUM_ITEMS = 1000
# Global variable to control number of splits
NUM_SPLITS = 4

# Initialize Whisper model globally
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16
model_id = "distil-whisper/distil-large-v3.5"

print("Loading Whisper model...")
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, cache_dir="cache"
)
whisper_model.to(device)
whisper_processor = AutoProcessor.from_pretrained(model_id)
print("Whisper model loaded successfully!")

# Global list to collect dataset entries
dataset_entries = []

def transcribe_audio(audio_data, sample_rate):
    """Transcribe audio data using Whisper"""
    target_sr = 16000
    if sample_rate != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
    
    # Prepare inputs for Whisper
    inputs = whisper_processor(audio_data, sampling_rate=target_sr, return_tensors="pt")
    inputs = {k: v.to(device).to(torch_dtype) if v.dtype == torch.float32 else v.to(device)
              for k, v in inputs.items()}

    # Create proper decoder input with Whisper special tokens
    decoder_start_token_id = whisper_model.config.decoder_start_token_id
    lang_token = whisper_processor.tokenizer.convert_tokens_to_ids("<|en|>")
    task_token = whisper_processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_token = whisper_processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

    decoder_input_ids = [decoder_start_token_id, lang_token, task_token, notimestamps_token]
    decoder_input_ids = torch.tensor([decoder_input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        generated_ids = whisper_model.generate(
            inputs["input_features"],
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=100,
            temperature=0.2,
            do_sample=False,
            pad_token_id=whisper_processor.tokenizer.pad_token_id,
            eos_token_id=whisper_processor.tokenizer.eos_token_id
        )

    generated_text = whisper_processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()

def split_and_save_audio(audio_array, sample_rate, output_dir, file_index):
    """Split audio into n overlapping segments and save files with transcription"""
    audio_length = len(audio_array)
    
    # Create n split points evenly distributed across the audio
    split_points = []
    for i in range(1, NUM_SPLITS):
        split_point = int((i / NUM_SPLITS) * audio_length)
        split_points.append(split_point)
    
    # Choose random overlap duration (between 0.5 seconds and 1/4 of audio length)
    max_overlap_duration = (audio_length / sample_rate) / 4
    overlap_duration = random.uniform(0.5, max_overlap_duration)
    overlap_samples = int(overlap_duration * sample_rate)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Store transcriptions for comparison
    chunk_transcriptions = []
    
    # Create overlapping splits
    for i in range(NUM_SPLITS):
        if i == 0:
            # First split: from start to first split point + overlap
            start = 0
            end = min(split_points[0] + overlap_samples, audio_length)
        elif i == NUM_SPLITS - 1:
            # Last split: from last split point - overlap to end
            start = max(split_points[-1] - overlap_samples, 0)
            end = audio_length
        else:
            # Middle splits: from previous split point - overlap to next split point + overlap
            start = max(split_points[i-1] - overlap_samples, 0)
            end = min(split_points[i] + overlap_samples, audio_length)
        
        split_audio = audio_array[start:end]
        sf.write(os.path.join(output_dir, f"split_{i+1}_{file_index:03d}.wav"), split_audio, sample_rate)
        
        # Transcribe chunk
        chunk_transcription = transcribe_audio(split_audio, sample_rate)
        chunk_transcriptions.append(chunk_transcription)
        
        # Print start and end times for each chunk
        start_time = start / sample_rate
        end_time = end / sample_rate
        duration = (end - start) / sample_rate
        print(f"  Split {i+1}: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
        print(f"    Transcription: '{chunk_transcription}'")
    
    # Save full audio and transcribe
    sf.write(os.path.join(output_dir, f"full_{file_index:03d}.wav"), audio_array, sample_rate)
    full_transcription = transcribe_audio(audio_array, sample_rate)
    
    # Filter out samples with less than 4 words
    word_count = len(full_transcription.split())
    if word_count < 4:
        print(f"Filtering out audio {file_index}: transcription has only {word_count} words")
        return
    
    print(f"\nAudio {file_index} Transcription Comparison:")
    print(f"Full audio transcription: '{full_transcription}'")
    print(f"Saved audio {file_index}: {NUM_SPLITS} splits with overlap: {overlap_samples} samples ({overlap_duration:.2f}s)\n")
    
    # Add to dataset
    dataset_entries.append({
        "chunks": chunk_transcriptions,
        "full_text": full_transcription
    })

def should_filter_sample(batch):
    """Check if sample should be filtered out based on duration"""
    # Filter out samples shorter than 8 seconds
    audio_array = batch["mp3"]["array"][0]
    sample_rate = batch["mp3"]["sampling_rate"][0].item()
    duration = len(audio_array) / sample_rate
    if duration < 8.0:
        print("Duration:", duration)
        return True
    
    return False

def create_and_push_dataset():
    """Create dataset from collected transcriptions and push to Hub"""
    if not dataset_entries:
        print("No dataset entries to save!")
        return
    
    # Create dataset
    dataset = Dataset.from_list(dataset_entries)
    
    # Print dataset info
    print(f"\nCreated dataset with {len(dataset)} entries")
    print("Sample entry:", dataset[0])

    # # Save locally first
    # dataset.save_to_disk("text_deduplication_dataset")
    # print("Dataset saved locally to 'text_deduplication_dataset'")
    
    # Push to Hub
    try:
        dataset.push_to_hub("BarryFutureman/text-based-deduplication", private=False)
        print("Dataset successfully pushed to Hub as 'text-based-deduplication'")
    except Exception as e:
        print(f"Error pushing to Hub: {e}")
        print("Make sure you're logged in with `huggingface-cli login`")

def main():
    # Load audiosnippets_long dataset in streaming mode
    dataset = load_dataset("mitermix/audiosnippets_long_2_5M", split="train", streaming=True, trust_remote_code=True)
    
    # Create dataloader with batch size 1
    dataloader = DataLoader(dataset, batch_size=1)
    
    # Output directory for saved audio files
    output_dir = "audio_splits"
    
    # Process items until we get NUM_ITEMS valid samples
    processed_count = 0
    
    # Create progress bar
    pbar = tqdm(total=NUM_ITEMS, desc="Processing audio samples")
    
    for i, batch in enumerate(dataloader):
        if processed_count >= NUM_ITEMS:
            break
            
        # Filter out unwanted samples
        if should_filter_sample(batch):
            continue

        # Extract audio data from batch - audio is under 'mp3' key
        audio_array = batch["mp3"]["array"][0].numpy()  # Get first item and convert to numpy
        sample_rate = batch["mp3"]["sampling_rate"][0].item()  # Get first item and convert to int
        
        # Split and save audio
        split_and_save_audio(audio_array, sample_rate, output_dir, processed_count + 1)
        processed_count += 1
        
        # Update progress bar
        pbar.update(1)
    
    pbar.close()
    
    print(f"Finished processing {processed_count} audio samples. Files saved in '{output_dir}' directory.")
    
    # Create and push dataset
    create_and_push_dataset()

if __name__ == "__main__":
    main()
