import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3.5"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, cache_dir="cache"
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Load audio file
audio_file_path = r"C:\Files\PythonProjects\TTS\AudioDataset\Detroit Become Human Kara Sounds\X0101K_ALICE_TENT_PC_X01KKARA_FRIENDLY_ENG.mp3"
audio_array, sample_rate = librosa.load(audio_file_path, sr=16000)

# Text prompt for conditioning
partial_text = "I'm not sure"

# Direct model inference with proper Whisper formatting
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
inputs = {k: v.to(device).to(torch_dtype) if v.dtype == torch.float32 else v.to(device) 
          for k, v in inputs.items()}

# Create proper decoder input with Whisper special tokens
decoder_start_token_id = model.config.decoder_start_token_id
lang_token = processor.tokenizer.convert_tokens_to_ids("<|en|>")
task_token = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
notimestamps_token = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

prompt_tokens = processor.tokenizer.encode(partial_text, add_special_tokens=False)
decoder_input_ids = [decoder_start_token_id, lang_token, task_token, notimestamps_token] + prompt_tokens
decoder_input_ids = torch.tensor([decoder_input_ids], dtype=torch.long).to(device)

with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_features"],
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id
    )

result_direct = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Combine partial text with generated result
full_transcription = partial_text + result_direct

# Print result
print("Partial text prompt:", partial_text)
print("Generated continuation:", result_direct)
print("Full transcription:", full_transcription)
