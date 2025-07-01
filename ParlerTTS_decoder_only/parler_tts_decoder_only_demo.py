from architecture.ParlerTTSDecoderOnly.modeling_parler_tts_decoder_only import *
from architecture.ParlerTTSDecoderOnly.configuration_parler_tts_decoder_only import ParlerTTSConfig
# from parler_tts.configuration_parler_tts import ParlerTTSConfig as OriginalParlerTTSConfig
from transformers import AutoTokenizer
import soundfile as sf
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

config = ParlerTTSConfig.from_pretrained("parler-tts/parler_tts_mini_v0.1", cache_dir="cache/models")
config.tie_word_embeddings = False
# config = ParlerTTSConfig(**config.to_dict())

model = ParlerTTSDecoderOnlyForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1",
                                                                     config=config,
                                                                     cache_dir="cache/models").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1", cache_dir="cache/models")

prompt = """
Why didn't the wife attend her husband's funeral? She wasn't much of a mourning person.
Why is being married worse than having to go to work? Because at least with work there's a chance you'll get a new boss.
I don't go to vampire weddings. They usually suck.
My husband is driving me to drink. I guess it's better than taking me for a walk."""

# input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
