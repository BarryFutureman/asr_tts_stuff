from flask import Flask, request, jsonify, Response
import base64
import io
from snac import SNAC
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import soundfile as sf
import librosa
import time
import threading
import queue
import json

app = Flask(__name__)


snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir="cache")
snac_model = snac_model.to("cpu")

model_name = "/student/jian1034/Desktop/TTS/orpheus-3b-0.1-kara-ft"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir="cache", device_map="auto")
model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")

chosen_voice = "kara"

class AudioTokensStreamer(threading.Thread):
	"""
	Streams tokens from model.generate and puts decode jobs into a queue.
	"""
	def __init__(self, decode_jobs_queue, min_tokens=27):
		super().__init__()
		self.token_queue = queue.Queue()
		self.stop_signal = object()
		self.decode_jobs_queue = decode_jobs_queue
		self.min_tokens = min_tokens
		self.tokens = []
		self.count = 0
		self.started = False
		self.buffer = []
		self.special_token = 128257
		self.remove_token = 128258
		self.offset = 128266
		self.last_special_idx = -1
		self.daemon = True

	def put(self, value):
		# Ignore if value is a list/tuple or a tensor with more than one element
		if isinstance(value, (list, tuple)):
			return
		if isinstance(value, torch.Tensor) and value.numel() > 1:
			return
		# Convert tensor to int if needed
		if isinstance(value, torch.Tensor):
			value = value.item()
		self.buffer.append(value)
		if value == self.special_token:
			self.last_special_idx = len(self.buffer) - 1
		self.token_queue.put(value)

	def end(self):
		self.token_queue.put(self.stop_signal)

	def run(self):
		while True:
			token = self.token_queue.get()
			if token is self.stop_signal:
				self.decode_jobs_queue.put("[STOP]")
				break
			# Only start after last occurrence of special token
			if not self.started:
				# Check if we've seen the last special token
				if self.token_queue.qsize() == 0:  # All tokens buffered
					if self.last_special_idx >= 0:
						self.tokens = []
						self.started = True
						# Remove tokens up to and including last_special_idx
						self.buffer = self.buffer[self.last_special_idx+1:]
						# Remove unwanted tokens and subtract offset
						self.buffer = [t - self.offset for t in self.buffer if t != self.remove_token]
						self.tokens.extend(self.buffer)
						self.count = len(self.tokens)
						self.buffer = []
			else:
				# Remove unwanted tokens and subtract offset
				if token != self.remove_token:
					self.tokens.append(token - self.offset)
					self.count += 1

			# Chunks of size min_tokens
			if self.count % 7 == 0 and self.count > self.min_tokens:
				buffer_to_proc = self.tokens
				self.tokens = []
				self.count = 0
				self.decode_jobs_queue.put(buffer_to_proc)

def run_decode(decode_jobs_queue, snac_model):
	"""
	Collects all available decode jobs, combines them, decodes at once, and returns audio.
	"""
	code_lists = []
	stop_signal = False
	
	# Blocks here
	time.sleep(2)
	job = decode_jobs_queue.get()
	if job == "[STOP]":
		stop_signal = True
		return None, stop_signal
	else:
		code_lists.append(job)

	
	# Collect all jobs currently available
	while True:
		try:
			job = decode_jobs_queue.get_nowait()
			if job == "[STOP]":
				stop_signal = True
				break
			else:
				code_lists.append(job)
		except queue.Empty:
			break
			
	if not code_lists:
		return None, stop_signal
	
	# Flatten code_lists
	combined = []
	for codes in code_lists:
		combined.extend(codes)
	print(f"Received {len(combined)} tokens for decoding.")

	# ...redistribute_codes logic from before...
	layer_1 = []
	layer_2 = []
	layer_3 = []
	for i in range((len(combined)+1)//7):
		layer_1.append(combined[7*i])
		layer_2.append(combined[7*i+1]-4096)
		layer_3.append(combined[7*i+2]-(2*4096))
		layer_3.append(combined[7*i+3]-(3*4096))
		layer_2.append(combined[7*i+4]-(4*4096))
		layer_3.append(combined[7*i+5]-(5*4096))
		layer_3.append(combined[7*i+6]-(6*4096))
	codes = [torch.tensor(layer_1).unsqueeze(0),
			 torch.tensor(layer_2).unsqueeze(0),
			 torch.tensor(layer_3).unsqueeze(0)]
	audio_hat = snac_model.decode(codes)
	audio_np = audio_hat.detach().squeeze().to("cpu").numpy()
	buf = io.BytesIO()
	sf.write(buf, audio_np, 24000, format='WAV')
	buf.seek(0)
	audio_base64 = base64.b64encode(buf.read()).decode('utf-8')
 
	return audio_base64, stop_signal

def generate_audio(text):
	prompt = f"{chosen_voice}: {text}"
	input_ids = tokenizer(prompt, return_tensors="pt").input_ids

	start_token = torch.tensor([[128259]], dtype=torch.int64)
	end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

	modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

	max_length = modified_input_ids.shape[1]
	padded_tensor = modified_input_ids
	attention_mask = torch.ones((1, max_length), dtype=torch.int64)

	input_ids_cuda = padded_tensor.to("cuda")
	attention_mask_cuda = attention_mask.to("cuda")

	decode_jobs_queue = queue.Queue()
	streamer = AudioTokensStreamer(decode_jobs_queue)
	streamer.start()

	def run_generate():
		with torch.no_grad():
			model.generate(
				input_ids=input_ids_cuda,
				attention_mask=attention_mask_cuda,
				max_new_tokens=2048,
				do_sample=True,
				temperature=0.6,
				top_p=0.95,
				repetition_penalty=1.1,
				num_return_sequences=1,
				eos_token_id=128258,
				streamer=streamer,
				output_scores=False,
				return_dict_in_generate=False,
			)
	thread = threading.Thread(target=run_generate)
	thread.start()

	stop = False
	while not stop:
		audio_base64, stop = run_decode(decode_jobs_queue, snac_model)
		if audio_base64 is not None:
			yield audio_base64

	thread.join()
	streamer.join()

@app.route('/tts', methods=['POST'])
def tts():
	data = request.get_json()
	if not data or 'text' not in data:
		return jsonify({'error': 'Missing "text" in request'}), 400
	text = data['text']

	def audio_stream():
		try:
			for audio_base64 in generate_audio(text):
				yield f'{json.dumps({"audio": audio_base64})}\n'
		except Exception as e:
			yield f'{json.dumps({"error": str(e)})}\n'

	return Response(audio_stream(), mimetype='application/json')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=7669)


