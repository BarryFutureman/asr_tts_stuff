from snac import SNAC
import torch
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import soundfile as sf
import librosa
from huggingface_hub import login, snapshot_download
import time

login(token="")

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir="cache")
snac_model = snac_model.to("cpu")

tokeniser_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "/student/jian1034/Desktop/TTS/orpheus-3b-0.1-kara-ft"  # "canopylabs/orpheus-3b-0.1-ft"

# # Download only model config and safetensors
# model_path = snapshot_download(
#     repo_id=model_name,
#     allow_patterns=[
#         "config.json",
#         "*.safetensors",
#         "model.safetensors.index.json",
#     ],
#     ignore_patterns=[
#         "optimizer.pt",
#         "pytorch_model.bin",
#         "training_args.bin",
#         "scheduler.pt",
#         "tokenizer.json",
#         "tokenizer_config.json",
#         "special_tokens_map.json",
#         "vocab.json",
#         "merges.txt",
#         "tokenizer.*"
#     ]
# )

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir="cache", device_map="auto")
model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")


# prompts = [
#     "Hey there my name is Tara, <chuckle> and I'm a speech generation model that can sound like a person.",
#     # "I've also been taught to understand and produce paralinguistic things like sighing, or chuckling, or yawning!",
#     "I live in San Francisco, and have, uhm let's see, 3 billion 7 hundred ... well, lets just say a lot of parameters.",
#     "I asked my wife if she ever fantasizes about me, and she said yes – about me taking out the trash, mowing the lawn, and doing the dishes. <chuckle>",
#     "I love to play games, and I can even play chess with you! <scream>",
#     "<sigh/> He qualified for the national tournament. <normal>",
#     "The, uhm, men at those, <chuckle>, fundraisers are always SO serious. <sad>",
#     "She died in a car accident, and I was so sad. <sad>"
# ]
# prompts = [
#     "Hey there my name is Barry, <chuckle> and I'm a speech generation model that can sound like a person.",
#     "I've also been taught to understand and produce paralinguistic things like sighing, or chuckling <chuckle>, or yawning! <yawn>",
#     "I live in Toronto, and have, uhm let's see, 3 billion 7 hundred ... well, lets just say a lot of parameters.",
#     "<angry> He qualified for the national tournament. ",
#     "<slow> He qualified for the national tournament. ",
#     "<happy> He qualified for the national tournament. ",
#     "<shout> He qualified for the national tournament. ",
#     "<crying> He qualified for the national tournament. ",
#     "<sleepy> He qualified for the national tournament.",
#     "<normal> <sigh/> He qualified for the national tournament.",
#     "<normal> The, uhm, men at those, <chuckle>, fundraisers are always SO serious. ",
# ]

prompts = [
  "<happy> Today is a wonderful day! I feel so joyful and grateful for everything.",
  "<sad> I just heard some bad news, and it makes me feel really down. <crying> I don't think I can handle this...",
  "<happy/> <happy> I just heard some bad news, <happy> and it makes me feel really down. <chuckle> I don't think I can handle this... <happy>",
  "<angry> Why did this have to happen? This is so frustrating!",
  "<fear> I'm really nervous about what might happen next. It's a bit scary.",
  "<disgust> That was absolutely revolting. I can't believe anyone would do that.",
  "<happy> That was absolutely revolting. <laugh> I can't believe anyone would do that.",
  "<surprise/> Wow! I did not expect that at all. What a shock!",
  "<fear> Today is a wonderful day! I feel so joyful and grateful for everything.",
  "<neutral> It's just another ordinary day, nothing special or unusual.",
  "<contempt> I can't believe they would do something so ridiculous. It's just absurd.",
]


# Uncomment the following lines to use a specific voice
chosen_voice = "KARA"
prompts = [f"{chosen_voice}: " + p for p in prompts]

all_input_ids = []

for prompt in prompts:
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids
  all_input_ids.append(input_ids)

start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human

all_modified_input_ids = []
for input_ids in all_input_ids:
  modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
  all_modified_input_ids.append(modified_input_ids)

all_padded_tensors = []
all_attention_masks = []
max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
for modified_input_ids in all_modified_input_ids:
  padding = max_length - modified_input_ids.shape[1]
  padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
  attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
  all_padded_tensors.append(padded_tensor)
  all_attention_masks.append(attention_mask)

all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
all_attention_masks = torch.cat(all_attention_masks, dim=0)

input_ids = all_padded_tensors.to("cuda")
attention_mask = all_attention_masks.to("cuda")

print("*** Model.generate is slow - see vllm implementation on github for realtime streaming and inference")
print("*** Increase/decrease inference params for more expressive less stable generations")

start_time = time.time()
with torch.no_grad():
  generated_ids = model.generate(
      input_ids=input_ids,
      attention_mask=attention_mask,
      max_new_tokens=2048,
      do_sample=True,
      temperature=0.6,
      top_p=0.95,
      repetition_penalty=1.1,
      num_return_sequences=1,
      eos_token_id=128258,
  )
end_time = time.time()
print(f"Generation took {end_time - start_time:.2f} seconds")
  
token_to_find = 128257
token_to_remove = 128258

token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

if len(token_indices[1]) > 0:
    last_occurrence_idx = token_indices[1][-1].item()
    cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
else:
    cropped_tensor = generated_ids

mask = cropped_tensor != token_to_remove

processed_rows = []

for row in cropped_tensor:
    masked_row = row[row != token_to_remove]
    processed_rows.append(masked_row)

code_lists = []

for row in processed_rows:
    row_length = row.size(0)
    new_length = (row_length // 7) * 7
    trimmed_row = row[:new_length]
    trimmed_row = [t - 128266 for t in trimmed_row]
    code_lists.append(trimmed_row)


def redistribute_codes(code_list):
  layer_1 = []
  layer_2 = []
  layer_3 = []
  for i in range((len(code_list)+1)//7):
    layer_1.append(code_list[7*i])
    layer_2.append(code_list[7*i+1]-4096)
    layer_3.append(code_list[7*i+2]-(2*4096))
    layer_3.append(code_list[7*i+3]-(3*4096))
    layer_2.append(code_list[7*i+4]-(4*4096))
    layer_3.append(code_list[7*i+5]-(5*4096))
    layer_3.append(code_list[7*i+6]-(6*4096))
  codes = [torch.tensor(layer_1).unsqueeze(0),
         torch.tensor(layer_2).unsqueeze(0),
         torch.tensor(layer_3).unsqueeze(0)]
  audio_hat = snac_model.decode(codes)
  return audio_hat

my_samples = []
for code_list in code_lists:
  samples = redistribute_codes(code_list)
  my_samples.append(samples)

# Save audio samples as WAV files
import re
for i, samples in enumerate(my_samples):
    audio_np = samples.detach().squeeze().to("cpu").numpy()
    emotion = re.sub(r'[^a-zA-Z]', '', prompts[i][6:24])
    sf.write(f"output_{emotion}_{i}.wav", audio_np, 24000)


