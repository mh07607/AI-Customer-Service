import pyaudio
import wave
import io
import threading
import time
import numpy as np
import keyboard
from typing import BinaryIO, Any
from google.genai import types
from constants import FORMAT, CHANNELS, RECORD_RATE, RECEIVE_SAMPLE_RATE, CHUNK, SILENCE_THRESHOLD, SILENCE_DURATION, client

def save_wave_file(filename, data, sample_rate=RECORD_RATE):
    """Save audio data to a WAV file"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for paInt16
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(data))

def play_audio(audio_data, sample_rate=RECEIVE_SAMPLE_RATE):
    """Play audio from binary data"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        output=True
    )
    
    print("Playing AI response...")
    stream.write(audio_data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio playback finished.")

def is_silent(data, threshold=SILENCE_THRESHOLD):
    """Check if the audio chunk is silent"""
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.mean(np.abs(audio_data)) < threshold

def record_audio():
    """Record audio from microphone until silence is detected or space is pressed"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RECORD_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("Recording... (Press SPACE to stop recording manually)")
    frames = []
    silent_chunks = 0
    silent_chunks_threshold = int(SILENCE_DURATION * RECORD_RATE / CHUNK)
    
    recording = True
    stop_event = threading.Event()
    
    # Start a thread to monitor for space key press
    def check_stop_key():
        while recording:
            if keyboard.is_pressed('space'):
                print("Space pressed, stopping recording...")
                stop_event.set()
                break
            time.sleep(0.1)
    
    key_thread = threading.Thread(target=check_stop_key)
    key_thread.daemon = True
    key_thread.start()
    
    try:
        while recording:
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Check for silence to auto-stop
            if is_silent(data):
                silent_chunks += 1
                if silent_chunks >= silent_chunks_threshold:
                    print("Silence detected, stopping recording...")
                    break
            else:
                silent_chunks = 0
                
            # Check if space was pressed
            if stop_event.is_set():
                break
    finally:
        recording = False
        stream.stop_stream()
        stream.close()
        p.terminate()

    return frames
    



def transcribe_audio(audio_frames):
    # Save audio to a WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RECORD_RATE)
        wf.writeframes(b''.join(audio_frames))
    buffer.seek(0)

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            'Transcribe the following audio to text:',
            types.Part.from_bytes(
                data=buffer.read(),
                mime_type='audio/wav',
            )
        ]
    )

    return response.text
