#!/usr/bin/env python3

import os
import subprocess
import time

# Voice model selection - Choose your meme voice!
AVAILABLE_VOICES = {
    'ryan': './voices/en_US-ryan-medium.onnx',          # Male voice - meme potential
    'lessac': './voices/en_US-lessac-medium.onnx',     # Original female voice  
    'robot': './voices/en_US-libritts_r-medium.onnx',  # Synthetic/robotic voice - BEST FOR MEMES
}

def test_voice(voice_name, message):
    """Test a specific voice with a message"""
    if voice_name not in AVAILABLE_VOICES:
        print(f"❌ Voice '{voice_name}' not available")
        return False
    
    voice_path = AVAILABLE_VOICES[voice_name]
    if not os.path.exists(voice_path):
        print(f"❌ Voice file not found: {voice_path}")
        return False
    
    piper_path = './piper/piper'
    if not os.path.exists(piper_path):
        print(f"❌ Piper TTS not found: {piper_path}")
        return False
    
    print(f"🎤 Testing {voice_name} voice: '{message}'")
    
    try:
        # Create temporary audio file
        temp_audio = "/tmp/voice_test.wav"
        
        # Generate speech with Piper
        process = subprocess.run([
            piper_path,
            '--model', voice_path,
            '--output_file', temp_audio
        ], input=message, text=True, capture_output=True, timeout=10)
        
        if process.returncode == 0:
            print(f"✅ Audio generated successfully")
            # Play the generated audio
            play_process = subprocess.run(['aplay', temp_audio], capture_output=True)
            if play_process.returncode == 0:
                print(f"✅ Audio played successfully with {voice_name} voice!")
            else:
                print(f"⚠️ Audio generated but failed to play: {play_process.stderr.decode()}")
            
            # Clean up temp file
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return True
        else:
            print(f"❌ Piper TTS failed: {process.stderr.decode()}")
            return False
    except Exception as e:
        print(f"❌ Error testing voice: {e}")
        return False

def main():
    print("🤖 Testing Robot Voice System")
    print("=" * 40)
    
    # Test messages
    test_messages = [
        "big black LINE detected... just how i like it",
        "i want that big Black line back daddy... pleaseeeee",
        "Obstacle detected! Turning around completely.",
        "shutting down, come with bigger black next time."
    ]
    
    # Test each voice
    for voice_name in ['robot', 'ryan', 'lessac']:
        print(f"\n🎵 Testing {voice_name.upper()} voice:")
        print("-" * 30)
        
        if voice_name in AVAILABLE_VOICES:
            for i, message in enumerate(test_messages):
                print(f"\nTest {i+1}: ", end="")
                test_voice(voice_name, message)
                time.sleep(1)  # Brief pause between tests
        else:
            print(f"❌ {voice_name} voice not configured")
    
    print("\n🎉 Voice testing complete!")
    print("\nTo use your preferred voice in the robot:")
    print("1. Edit main.py")
    print("2. Change PREFERRED_VOICE = 'robot'  # or 'ryan', 'lessac'")
    print("3. Run the robot with: python main.py")

if __name__ == "__main__":
    main() 