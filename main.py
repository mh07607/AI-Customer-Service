from google.genai import types
from constants import client
from audio_functions import play_audio, record_audio, transcribe_audio


def chat_with_ai():
    """Main conversation loop with the AI"""    
    
    print("\n--- AI Customer Service Conversation System ---")
    print("Press SPACE when you're done speaking or wait for silence detection")
    print("Type 'exit' at any time to end the conversation\n")
    
    chat = client.chats.create(model="gemini-2.0-flash")
    # First AI response to start the conversation
    response = chat.send_message(        
        "You are a helpful customer service assistant for a bank. Keep your responses concise and conversational. If a customer makes a specific request, try to help or direct them to the appropriate resource. Start by greeting the customer and asking how you can help."
    )
    
    # Get audio for the initial greeting
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash-preview-tts",
        contents=response.text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name='Kore',
                    )
                )
            ),
        )
    ):
        ai_audio_data = chunk.candidates[0].content.parts[0].inline_data.data
        play_audio(ai_audio_data)
        
    
    # Play the AI's greeting
    # print(f"AI: {response.text}")
    # ai_audio_data = audio_response.candidates[0].content.parts[0].inline_data.data
    # play_audio(ai_audio_data)
    
    # Main conversation loop
    while True:
        # Record user audio
        print("\nYour turn (recording...):")
        audio_frames = record_audio()
        
        if not audio_frames:
            print("No audio recorded. Please try again.")
            continue
        
        transcribed_text = transcribe_audio(audio_frames)

        # Get AI text response
        response = chat.send_message(transcribed_text)
        
        print(f"AI: {response.text}")
        
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash-preview-tts",
            contents=response.text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='Kore',
                        )
                    )
                ),
            )
        ):
            ai_audio_data = chunk.candidates[0].content.parts[0].inline_data.data
            play_audio(ai_audio_data)
# Run the conversation system
if __name__ == "__main__":
    chat_with_ai()