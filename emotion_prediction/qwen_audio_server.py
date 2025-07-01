from flask import Flask, request, jsonify
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import librosa
from io import BytesIO
from urllib.request import urlopen

app = Flask(__name__)

# Initialize the processor and model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir="cache")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct",
                                                           torch_dtype=torch.float16,
                                                           #    load_in_8bits=True,
                                                           device_map="auto", cache_dir="cache")


@app.route('/process_audio', methods=['POST'])
def process_audio():
    conversation = request.json

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(
                            BytesIO(urlopen(ele['audio_url']).read()),
                            sr=processor.feature_extractor.sampling_rate)[0]
                    )
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True).to("cuda")

    generate_ids = model.generate(**inputs, max_length=1024)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5069)
