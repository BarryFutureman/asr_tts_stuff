from flask import Flask, request, jsonify
import base64
import io
from snac import SNAC
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import soundfile as sf
import librosa
import time

app = Flask(__name__)


snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir="cache")
snac_model = snac_model.to("cpu")

model_name = "/student/jian1034/Desktop/TTS/orpheus-3b-0.1-kara-ft"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir="cache", device_map="auto")
model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")

chosen_voice = "kara"

def generate_audio(text):
    prompt = f"{chosen_voice}: {text}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

    max_length = modified_input_ids.shape[1]
    padding = 0  # Only one prompt at a time
    padded_tensor = modified_input_ids
    attention_mask = torch.ones((1, max_length), dtype=torch.int64)

    input_ids_cuda = padded_tensor.to("cuda")
    attention_mask_cuda = attention_mask.to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids_cuda,
            attention_mask=attention_mask_cuda,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258,
        )

    token_to_find = 128257
    token_to_remove = 128258

    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = generated_ids

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

    samples = redistribute_codes(code_lists[0])
    audio_np = samples.detach().squeeze().to("cpu").numpy()
    # Write to buffer
    buf = io.BytesIO()
    sf.write(buf, audio_np, 24000, format='WAV')
    buf.seek(0)
    audio_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return audio_base64

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request'}), 400
    text = data['text']
    try:
        audio_base64 = generate_audio(text)
        return jsonify({'audio': audio_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7669)


