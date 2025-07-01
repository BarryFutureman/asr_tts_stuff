
import requests
import torch
import soundfile
from PIL import Image
import soundfile
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig,pipeline,AutoTokenizer

model_path = 'microsoft/Phi-4-multimodal-instruct'

kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, cache_dir='cache')

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='bfloat16',
    device_map='auto',
    cache_dir='cache',
    _attn_implementation='sdpa',
).cuda()

generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json', cache_dir='cache')

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

speech_prompt = "Transcribe the attached audio. Available tags for speech: <chuckle>, <scream>, <laugh>, etc. Example: <happy> I, <chuckle> am a butterfly!"
prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'

audio = soundfile.read('/student/jian1034/Desktop/TTS/output_0_kara_tara.wav')

inputs = processor(text=prompt, audios=[audio], return_tensors='pt').to('cuda:0')

generate_ids = model.generate(
    **inputs,
    max_new_tokens=1200,
    generation_config=generation_config,
)

generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]

response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(response)

