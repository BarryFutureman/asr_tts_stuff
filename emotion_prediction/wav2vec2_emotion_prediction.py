import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Load model and processor
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Define emotion labels
emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


def preprocess_audio(file_path, target_sr=16000):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=target_sr)

    # Convert to tensor
    input_values = processor(audio, return_tensors="pt", sampling_rate=target_sr).input_values
    return input_values


audio_path = "C:\Files\PythonProjects\TTS\ParlerTTS\parler_tts_out.wav"
input_values = preprocess_audio(audio_path)

with torch.no_grad():
    logits = model(input_values).logits

# Get predicted emotion
predicted_class = torch.argmax(logits, dim=-1).item()
predicted_emotion = emotions[predicted_class]

print(f"Predicted Emotion: {predicted_emotion}")
