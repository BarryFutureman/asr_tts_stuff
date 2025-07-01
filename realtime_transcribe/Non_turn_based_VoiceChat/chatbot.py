import requests
import json
from datetime import datetime, timedelta
import threading
import queue
import base64
import re
import time

USER_NAME = "Barry"
USER_PROMPT_TEMPLATE = f"{USER_NAME}: "
CHATBOT_NAME = "Kara"
SYSTEM_PROMPT = f'''You are {CHATBOT_NAME}, You are a witty character who loves to engage in fun and lively conversations. Your responses are short and to the point, conversational.
Available tags for speech: <happy>, <sad>, <angry>, <fear>, <disgust>, <contempt>, <surprise>, <neutral>, <chuckle>, <gasp>, <shout>, <whisper>
'''
USER_SYSTEM_PROMPT = ""

TURN_DETECTION_PROMPT = """You are a turn detection system. Analyze the conversation context and current transcription to determine if the assistant should respond now.

Rules:
- Respond with only "YES" or "NO"
- YES: If the user has finished a complete thought, asked a question, or there's a natural pause for response
- NO: If the user is still speaking, incomplete sentence, or just started talking
- Consider conversation flow and context

Current transcription: "{transcription}"
Context: This is real-time speech transcription. Fixed text means the user has paused/finished that part."""


class ChatbotSettings:
    def __init__(self, ):
        self.chatbot_name = CHATBOT_NAME
        self.system_prompt = SYSTEM_PROMPT
        self.user_name = USER_NAME
        self.user_system_prompt = USER_SYSTEM_PROMPT


class Chatbot:
    def __init__(self, use_local=True, realtime_voice_processor=None):
        if use_local:
            self.api_key = "dummy-key"  # Local server doesn't need real key
            self.base_url = "http://142.1.46.70:8000"
            self.model = "dummy"
        else:
            self.api_key = ""
            self.base_url = "https://api.groq.com/openai/v1"
            self.model = "llama-3.3-70b-versatile"

        # Build prompt
        self.system_messages = []
        self.messages = []
        self.settings = ChatbotSettings()
        self.tts = TextToSpeech()
        self.result_queue = queue.Queue()
        
        # Add transcription tracking for fallback
        self.last_transcription = ""
        self.last_transcription_time = time.time()
        self.transcription_stable_threshold = 1.0  # seconds

        if realtime_voice_processor:
            self.realtime_voice_processor = realtime_voice_processor
            self.run_thread = threading.Thread(target=self.run)
            self.run_thread.daemon = True
            self.run_thread.start()

    def run(self):
        while True:
            try:
                if self.realtime_voice_processor and self.should_respond_llm():
                    # Get the full transcription before flushing
                    full_transcription = self.realtime_voice_processor.get_full_text().strip()
                    
                    if full_transcription:
                        # Flush fixed chunks and audio buffer to prevent repetition
                        self.realtime_voice_processor.flush()
                        
                        # Create new message from user transcription
                        new_messages = [{"role": "user", "content": full_transcription}]
                        
                        # Generate and handle chatbot response
                        for response_chunk in self.chat(new_messages):
                            if response_chunk["text"]:
                                print(f"\033[96m{response_chunk['text']}\033[0m", end="", flush=True)
                                self.result_queue.put(response_chunk["text"])
                            # Handle audio if available
                            if response_chunk["audio"]:
                                # Audio handling would go here
                                pass
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                print(f"Error in chatbot run loop: {e}")
                time.sleep(1)

    def should_respond_llm(self):
        """Use LLM completion to determine if the chatbot should respond based on conversation context"""
        if not self.realtime_voice_processor or not self.realtime_voice_processor.full_running_transcription:
            return False
        
        current_transcription = self.realtime_voice_processor.get_full_text().strip()
        current_time = time.time()
        
        # Fallback: Check if transcription has stopped changing
        if current_transcription != self.last_transcription:
            self.last_transcription = current_transcription
            self.last_transcription_time = current_time
        elif current_transcription and (current_time - self.last_transcription_time) >= self.transcription_stable_threshold:
            print(f"\n[Fallback] Transcription stable for {self.transcription_stable_threshold}s -> RESPOND\n")
            return True
        
        # If transcription is empty or just changed, don't respond yet
        if not current_transcription or (current_time - self.last_transcription_time) < 0.5:
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            url = f"{self.base_url}/chat/completions"
            
            # Build conversation context with current transcription
            messages = []
            
            # Add recent conversation history for context
            if len(self.messages) >= 4:
                # Add <start> tags to recent messages for consistency
                recent_messages = self.messages[-4:]
                for msg in recent_messages:
                    content_with_start = f"<start>{msg['content']}<end>"
                    messages.append({"role": msg["role"], "content": content_with_start})
            
            # Add current user transcription with <start> tag
            messages.append({"role": "user", "content": f"<start>{self.settings.user_name}: {current_transcription}"})
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 24,
                "stream": False,
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            completion = result["choices"][0]["message"]["content"].strip()
            
            # Analyze completion to determine turn
            should_respond = self._analyze_completion(completion, current_transcription)
            
            print(f"\n[Turn Detection] Completion: '{completion[:30]}...' -> {'RESPOND' if should_respond else 'WAIT'}\n")
            
            return should_respond
            
        except Exception as e:
            print(f"Turn detection error: {e}")
            # Fallback to simple rule-based detection
            return self.should_speak_fallback()
    
    def _analyze_completion(self, completion, current_transcription):
        """Analyze the model's completion to determine if user is done speaking"""
        # Simple check: if completion starts with newline or end token, user is done
        return completion.startswith('\n') or completion.startswith('<|im_end|>') or completion.startswith('<end>')

    def get_curr_date_time(self):
        utc_now = datetime.utcnow()
        est_now = utc_now - timedelta(hours=4)
        return est_now

    def chat(self, new_messages):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"

        # current_time = self.get_curr_date_time().strftime("%Y-%m-%d %H:%M:%S")
        # self.messages.append({"role": "user", "content": f"Current Time: {current_time} EST"})
        for msg in new_messages:
            self.messages.append({"role": msg["role"], "content": f"{self.settings.user_name}: {msg['content']}"})
            if msg["content"] == "clear":
                self.messages = []
        self.messages.append({"role": "assistant", "content": f"{self.settings.chatbot_name}: "})

        text_queue = queue.Queue()
        audio_queue = queue.Queue()

        def generate_text():
            data = {
                "model": self.model,
                "messages": self.build_messages(),
                "temperature": 0.9,
                "max_tokens": 2048,
                "stream": True
            }

            try:
                with requests.post(url, headers=headers, json=data, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            if decoded_line == "data: [DONE]":
                                break
                            data_json = json.loads(decoded_line.replace("data: ", ""))
                            if "choices" in data_json:
                                chunk = data_json["choices"][0]["delta"].get("content", "")
                                if chunk:
                                    text_queue.put(chunk)
            except requests.HTTPError as ex:
                error_content = ex.response.text if ex.response is not None else str(ex)
                text_queue.put(f"HTTP error occurred: {getattr(ex.response, 'status_code', 'N/A')} - {error_content}")
            finally:
                text_queue.put("[TEXT_END]")

        def generate_audio():
            full_response = []
            while True:
                chunk = text_queue.get()
                if chunk == "[TEXT_END]":
                    break
                full_response.append(chunk)
                # Yield text with no audio yet
                audio_queue.put({"text": chunk, "audio": None})
            # # After all text is received, call TTS and stream audio
            # response_text = ''.join(full_response)
            # for audio_chunk in self.tts.speak(response_text):
            #     # Each audio_chunk is a base64 string
            #     audio_queue.put({"text": None, "audio": audio_chunk})
            audio_queue.put({"text": "[AUDIO_END]", "audio": None})

        thread_text = threading.Thread(target=generate_text)
        thread_audio = threading.Thread(target=generate_audio)
        thread_text.start()
        thread_audio.start()

        while True:
            item = audio_queue.get()
            if item["text"] == "[AUDIO_END]":
                break

            if item["text"]:
                self.messages[-1]["content"] += item["text"]
            yield item

        thread_text.join()
        thread_audio.join()

    def build_messages(self):
        final_msgs = []

        # Start with the system prompt
        self.system_messages = [{"role": "system", "content": self.settings.system_prompt}]

        final_msgs.extend(self.system_messages)
        final_msgs.extend(self.messages)

        return final_msgs


class TextToSpeech:
    def __init__(self):
        self.api_key = ""
        self.base_url = "https://api.groq.com/openai/v1/audio/speech"

    def speak(self, text):
        # Remove content within backticks, angle brackets, square brackets, and asterisks
        clean_text = re.sub(r'`[^`]*`', '', text)  # Remove content within backticks
        clean_text = re.sub(r'<[^>]*>', '', clean_text)  # Remove content within angle brackets
        clean_text = re.sub(r'\[[^\]]*\]', '', clean_text)  # Remove content within square brackets
        clean_text = re.sub(r'\*[^*]*\*', '', clean_text)  # Remove content within asterisks
        clean_text = clean_text.strip()  # Remove leading/trailing whitespace
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "playai-tts",
            "input": clean_text,
            "voice": "Celeste-PlayAI",
            "response_format": "wav"
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            yield audio_base64
            
        except Exception as e:
            print(f"TTS error: {e}")
            return