import pyaudio
import wave
import librosa
import soundfile as sf
import io
import time
import threading
import keyboard
import asyncio
from google import genai
from google.genai import types
from constants import client

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2  # in seconds

def is_silent(data):
    """Return True if below silence threshold"""
    return max(data) < SILENCE_THRESHOLD

def record_audio():
    """Record audio until silence or space is pressed"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RECORD_RATE, input=True,
                    frames_per_buffer=CHUNK)

    print("Recording... (Press SPACE to stop manually)")
    frames = []
    silent_chunks = 0
    threshold_chunks = int(SILENCE_DURATION * RECORD_RATE / CHUNK)
    stop_event = threading.Event()

    def check_key():
        while True:
            if keyboard.is_pressed('space'):
                print("Stopped manually.")
                stop_event.set()
                break
            time.sleep(0.1)

    threading.Thread(target=check_key, daemon=True).start()

    try:
        while not stop_event.is_set():
            data = stream.read(CHUNK)
            frames.append(data)
            audio_data = wave.struct.unpack("%dh" % (len(data) // 2), data)
            if is_silent(audio_data):
                silent_chunks += 1
                if silent_chunks > threshold_chunks:
                    print("Stopped due to silence.")
                    break
            else:
                silent_chunks = 0
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    return b''.join(frames)

def save_pcm(raw_audio):
    """Convert raw audio to PCM .wav format"""
    filename = "temp_input.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RECORD_RATE)
        wf.writeframes(raw_audio)
    return filename

async def chat_with_gemini(audio_bytes):
    model = "gemini-2.0-flash-live-001"
    
    async with client.aio.live.connect(model=model, config={"response_modalities": ["TEXT"]}) as session:
        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )
        async for response in session.receive():
            if response.text:
                return response.text
    return "I couldn't understand you."

async def speak_with_gemini(text):
    model = "gemini-2.5-flash-preview-tts"
    
    async with client.aio.live.connect(model=model, config={"response_modalities": ["AUDIO"]}) as session:
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": text}]}, turn_complete=True
        )

        wf = wave.open("response.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        async for response in session.receive():
            if response.data:
                wf.writeframes(response.data)
        wf.close()

    # Play audio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=24000, output=True)
    wf = wave.open("response.wav", "rb")
    data = wf.readframes(CHUNK)
    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()

async def main():
    while True:
        raw_audio = record_audio()
        save_pcm(raw_audio)
        y, sr = librosa.load("temp_input.wav", sr=16000)
        buffer = io.BytesIO()
        sf.write(buffer, y, sr, format='RAW', subtype='PCM_16')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        text = await chat_with_gemini(audio_bytes)
        print(f"Gemini: {text}")
        await speak_with_gemini(text)

if __name__ == "__main__":
    asyncio.run(main())
