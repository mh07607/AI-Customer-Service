import pyaudio
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_AI_API_KEY")

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_RATE = 16000  # Standard for speech recognition
RECEIVE_SAMPLE_RATE = 24000  # Standard for Gemini TTS
CHUNK = 1024
SILENCE_THRESHOLD = 1000  # Adjust based on your microphone and environment
SILENCE_DURATION = 2.0  # Seconds of silence to end recording
# Set up Google Generative AI client
client = genai.Client(api_key=api_key)