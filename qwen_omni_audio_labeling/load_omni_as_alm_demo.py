from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import BitsAndBytesConfig
from qwen_omni_utils import process_mm_info


# You can directly insert a local file path, a URL, or a base64-encoded audio into the position where you want in the text.
# messages = [
#   # Audio
#     ## Local audio path
#     [{"role": "system", "content":[{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
#      {"role": "user", "content": [{"type": "audio", "audio": "giggle.wav"}, {"type": "text", "text": "Please describe this audio."}]}],
#     [{"role": "user", "content": [{"type": "audio", "audio": "laugh.wav"}, {"type": "text", "text": "Transcribe the audio. Avaliable tags: (angry) (sad) (excited) (surprised) (satisfied) (delighted) (scared) (worried) (upset) (nervous) (frustrated) (depressed)(empathetic) (embarrassed) (disgusted) (moved) (proud) (relaxed)(grateful) (confident) (interested) (curious) (confused) (joyful) (laughing) (chuckling) (sobbing) (crying loudly) (sighing) (panting) (groaning) (crowd laughing) (background laughter) (audience laughing) (in a hurry tone) (shouting) (screaming) (whispering) (soft tone) Output the thinking process (less than 50 words) in <think> </think> and final answer in <answer> </answer>."}]}],
#     [{"role": "user", "content": [{"type": "audio", "audio": "giggle.wav"}, {"type": "text", "text": "What animal is the main source of sound in the video? ['dog', 'wasp', 'honeybee', 'dragonfly'] Output the thinking process (less than 50 words) in <think> </think> and final answer in <answer> </answer>."}]}],
# ]
messages = [
  # Audio
    ## Local audio path
    [{"role": "system", "content":[{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
     {"role": "user", "content": [{"type": "audio", "audio": "jukebox.wav"}, {"type": "text", "text": "Transcribe the given audio. Avaliable tags: (angry) (sad) (excited) (surprised) (satisfied) (delighted) (scared) (worried) (upset) (nervous) (frustrated) (depressed)(empathetic) (embarrassed) (disgusted) (moved) (proud) (relaxed)(grateful) (confident) (interested) (curious) (confused) (joyful) (laughing) (chuckling) (sobbing) (crying loudly) (sighing) (panting) (groaning) (crowd laughing) (background laughter) (audience laughing) (in a hurry tone) (shouting) (screaming) (whispering) (soft tone). Output the thinking process (less than 50 words) in <think> </think> and final answer in <answer> </answer>."}]}],
]

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    '/student/jian1034/Desktop/TTS/OmniFinetune/result', 
    device_map="auto", 
    cache_dir="cache",
    quantization_config=quantization_config
)
print("here")
processor = Qwen2_5OmniProcessor.from_pretrained('KE-Team/Ke-Omni-R-3B', cache_dir="cache")

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
inputs = processor(text=text, images=images, videos=videos, audio=audios, padding=True, return_tensors="pt")

# Move inputs to the same device as the model
inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

generation = model.generate(**inputs, temperature=0.7, do_sample=True, max_new_tokens=64)
generated_ids = generation[:, inputs['input_ids'].size(1):]
completions = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(completions)
