from flask import Flask, request, jsonify
# from flask_cors import CORS
import wave
import io
import threading
import os
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
import time
import difflib
from chatbot import Chatbot

app = Flask(__name__)
# CORS(app)

# Global audio buffer
audio_buffer = bytearray()
buffer_lock = threading.Lock()

# Audio parameters (will be set from first received WAV)
audio_params = None

class ProcessorResult:
    def __init__(self, input_audio_buffer_size: int, text: str):
        self.input_audio_buffer_size = input_audio_buffer_size
        self.text = text


class TranscribedChunk:
    def __init__(self, text: str, is_fixed: bool = False):
        self.text = text
        self.is_fixed = is_fixed

    def __repr__(self):
        return f"{self.text}[{'fixed' if self.is_fixed else 'active'}]"

class RealtimeVoiceProcessor:
    def __init__(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.running = True
        
        # Initialize Whisper model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "distil-whisper/distil-large-v3.5"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, cache_dir="cache"
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Initialize Silero VAD
        self.silero_model, self.silero_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.silero_model.to("cpu")
        (self.get_speech_timestamps, _, self.read_audio, _, _) = self.silero_utils

        self.full_running_transcription = []  # List of TranscribedChunk objects
        self.last_processed_size = 0
        
        self.thread.start()

    def get_partial_text_context(self, remove_last_n_words=3):
        if not self.full_running_transcription:
            return "", "", ""

        # Get all text from chunks
        full_text_without_last_chunk = "".join(chunk.text for chunk in self.full_running_transcription[-4:-1])

        if self.full_running_transcription[-1].is_fixed is True:
            return full_text_without_last_chunk, "", ""

        last_chunk_words = self.full_running_transcription[-1].text.strip().split()

        partial_text = " ".join(last_chunk_words[:-remove_last_n_words])
        removed_words = " ".join(last_chunk_words[-remove_last_n_words:])
        full_partial_text = full_text_without_last_chunk + partial_text

        return full_partial_text, partial_text, removed_words

    def get_full_text(self):
        return "".join(chunk.text for chunk in self.full_running_transcription)

    def vad_step(self, audio_data):
        # Convert audio data to tensor for Silero VAD
        data = torch.tensor(audio_data, dtype=torch.float32)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(data, self.silero_model, return_seconds=True)
        print(speech_timestamps)
        
        # If no speech detected, return None
        if not speech_timestamps:
            return None
        
        # Find first start and last end timestamps
        first_start = speech_timestamps[0]["start"]
        last_end = speech_timestamps[-1]["end"]
        
        # Convert timestamps to sample indices (assuming 16kHz sample rate)
        sample_rate = 16000
        start_sample = int(first_start * sample_rate)
        end_sample = int(last_end * sample_rate)
        
        # Trim audio to speech segments
        trimmed_audio = audio_data[start_sample:end_sample]
        
        return trimmed_audio
    
    def detect_repetition(self, new_text, lookback_chunks=8):
        if not self.full_running_transcription or not new_text.strip():
            return False
        for lookback in range(2, lookback_chunks):
            # Get recent text from the last few chunks for comparison
            recent_chunks = self.full_running_transcription[-lookback:]
            recent_text = "".join(chunk.text for chunk in recent_chunks).strip()

            if not recent_text:
                return False

            new_text_clean = new_text.strip()

            # Check for exact substring match
            if new_text_clean in recent_text:
                return True

            # Check for high similarity using difflib
            def fuzzy_contains(long_string, short_string, threshold=0.8):
                """
                Check if short_string fuzzily appears in long_string using SequenceMatcher.
                Returns True if any substring of long_string has a similarity ratio above threshold.
                """
                len_short = len(short_string)
                for i in range(len(long_string) - len_short + 1):
                    window = long_string[i:i + len_short]
                    ratio = difflib.SequenceMatcher(None, window, short_string).ratio()
                    if ratio >= threshold:
                        return True
                return False

            if fuzzy_contains(recent_text, new_text_clean):
                return True
        
        return False
    
    def transcribe_audio(self, audio_data, sample_rate):
        """Transcribe audio data directly from numpy array"""
        target_sr = 16000
        if sample_rate != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)

        audio_data = audio_data[-target_sr * 16:]

        audio_data = self.vad_step(audio_data)
        if audio_data is None:
            return "..."

        # Get partial text context from existing transcription
        full_partial_text, part_partial_text, removed_words = self.get_partial_text_context(remove_last_n_words=8)
        partial_text = full_partial_text

        # Prepare inputs for Whisper
        inputs = self.processor(audio_data, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device).to(self.torch_dtype) if v.dtype == torch.float32 else v.to(self.device)
                  for k, v in inputs.items()}

        # Create proper decoder input with Whisper special tokens
        decoder_start_token_id = self.model.config.decoder_start_token_id
        lang_token = self.processor.tokenizer.convert_tokens_to_ids("<|en|>")
        task_token = self.processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_token = self.processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

        prompt_tokens = self.processor.tokenizer.encode(partial_text, add_special_tokens=False) if partial_text else []
        decoder_input_ids = [decoder_start_token_id, lang_token, task_token, notimestamps_token] + prompt_tokens
        decoder_input_ids = torch.tensor([decoder_input_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=16,
                # temperature=0.2,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )

        generated_text = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if not self.full_running_transcription:
            self.full_running_transcription.append(
                TranscribedChunk(generated_text[len(removed_words):], is_fixed=False))
        elif len(generated_text.split()) <= 3 and ("thank you" in generated_text.lower()
                                                   or generated_text.strip().startswith("I")
                                                   or generated_text.strip().startswith("Yeah")
                                                   or generated_text.strip().lower().startswith("you")
                                                   or generated_text.strip().startswith("Okay")):
            # Remove silence
            pass
        elif removed_words and generated_text.strip().startswith(removed_words.strip()):
            if generated_text.startswith(" "):
                removed_words = " " + removed_words.strip()

            self.full_running_transcription[-1].is_fixed = True
            self.full_running_transcription.append(
                TranscribedChunk(generated_text[len(removed_words):], is_fixed=False))

            if len(self.full_running_transcription) > 8:
                self.full_running_transcription[1].text = self.full_running_transcription[0].text + self.full_running_transcription[1].text
                self.full_running_transcription.pop(0)
        else:
            # Remove repetition
            is_repetition = self.detect_repetition(generated_text)

            if not is_repetition:
                self.full_running_transcription[-1].is_fixed = False
                self.full_running_transcription[-1].text = part_partial_text + generated_text

        # print(f"Partial context: '{partial_text}'")
        # print(f"Removed words: '{removed_words}'")
        # print(f"Generated: '{generated_text}'")
        # print(f"Full transcription: '{''.join([str(_) for _ in self.full_running_transcription])}'")
        
        return (
            f"## {self.get_full_text()}  \n"
            f"**Partial context:** `{partial_text}`  \n"
            f"**Removed words:** `{removed_words}`  \n"
            f"**Generated:** `{generated_text}`  \n"
            f"**Chunks:** `{''.join([str(_) for _ in self.full_running_transcription])}`"
        )

    def flush(self):
        """Remove all fixed chunks from transcription and clear audio buffer to prevent repetition"""
        global audio_buffer
        
        # Keep only non-fixed chunks
        self.full_running_transcription = [
            chunk for chunk in self.full_running_transcription 
            if not chunk.is_fixed
        ]
        
        # Clear the audio buffer to prevent reprocessing old audio
        with buffer_lock:
            audio_buffer.clear()
            
        # Reset the last processed size
        self.last_processed_size = 0
        
        print("Flushed fixed chunks and cleared audio buffer")

    def run(self):
        while self.running:
            try:
                # Check if there's new audio data
                with buffer_lock:
                    current_size = len(audio_buffer)
                    if current_size > self.last_processed_size and audio_params is not None:
                        # Copy current buffer
                        buffer_copy = bytes(audio_buffer)
                        params_copy = audio_params.copy()
                        self.last_processed_size = current_size
                    else:
                        buffer_copy = None
                        params_copy = None
                
                if buffer_copy and params_copy:
                    # Convert bytes to numpy array
                    if params_copy['sampwidth'] == 2:
                        audio_data = np.frombuffer(buffer_copy, dtype=np.int16)
                    elif params_copy['sampwidth'] == 4:
                        audio_data = np.frombuffer(buffer_copy, dtype=np.int32)
                    else:
                        time.sleep(0.1)
                        continue
                    
                    # Convert to float32 and normalize
                    audio_data = audio_data.astype(np.float32)
                    if params_copy['sampwidth'] == 2:
                        audio_data /= 32768.0
                    elif params_copy['sampwidth'] == 4:
                        audio_data /= 2147483648.0
                    
                    # Handle stereo audio by taking first channel
                    if params_copy['channels'] == 2:
                        audio_data = audio_data[::2]
                    
                    # Transcribe the audio
                    transcribed_text = self.transcribe_audio(audio_data, params_copy['framerate'])
                    print(f"Transcription result: {transcribed_text}")
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in processor run loop: {e}")
                time.sleep(1)

# Initialize processor
voice_processor = RealtimeVoiceProcessor()
chatbot_instance = Chatbot(realtime_voice_processor=voice_processor)

@app.route('/stream_audio', methods=['POST'])
def stream_audio():
    global audio_buffer, audio_params
    
    try:
        # Get the audio data from request
        audio_data = request.data
        
        if not audio_data:
            return jsonify({'error': 'No audio data received'}), 400
        
        # Read WAV data
        wav_io = io.BytesIO(audio_data)
        
        with wave.open(wav_io, 'rb') as wav_file:
            # Store audio parameters from first WAV file
            if audio_params is None:
                audio_params = {
                    'channels': wav_file.getnchannels(),
                    'sampwidth': wav_file.getsampwidth(),
                    'framerate': wav_file.getframerate()
                }
            
            # Read audio frames
            frames = wav_file.readframes(wav_file.getnframes())
            
            # Thread-safe buffer update
            with buffer_lock:
                audio_buffer.extend(frames)
                
            # Update output file
            update_output_file()
            
        return jsonify({'status': 'success', 'buffer_size': len(audio_buffer)}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def update_output_file():
    """Update the output.wav file with current buffer content"""
    global audio_buffer, audio_params
    
    if audio_params is None or len(audio_buffer) == 0:
        return
    
    try:
        with buffer_lock:
            # Create a copy of buffer for file writing
            buffer_copy = bytes(audio_buffer)
        
        # Write to output.wav
        with wave.open('output.wav', 'wb') as output_file:
            output_file.setnchannels(audio_params['channels'])
            output_file.setsampwidth(audio_params['sampwidth'])
            output_file.setframerate(audio_params['framerate'])
            output_file.writeframes(buffer_copy)
            
    except Exception as e:
        print(f"Error updating output file: {e}")

@app.route('/transcription', methods=['GET'])
def get_transcription():
    """Get the current transcription"""
    try:
        return jsonify({
            'transcription': voice_processor.get_full_text(),
            'chunks': [{'text': chunk.text, 'is_fixed': chunk.is_fixed} for chunk in voice_processor.full_running_transcription]
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting audio streaming server...")
    app.run(host='0.0.0.0', port=5669, debug=False)
