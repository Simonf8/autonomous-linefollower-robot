#!/usr/bin/env python3

import os
import subprocess
import time
import tempfile
import random

class MemeVoiceSystem:
    def __init__(self):
        self.voices = {
            # ACTUAL MEME VOICES with eSpeak parameters
            'trump': {
                'engine': 'espeak',
                'voice': 'en+m3',           # Male voice
                'speed': '160',             # Slightly faster
                'pitch': '40',              # Lower pitch, authoritative
                'amplitude': '200',         # Louder
                'gap': '5',                 # Slight gaps between words
                'variant': 'm3',            # Male variant 3
                'prefix': "Folks, let me tell you, ",
                'suffix': " It's gonna be tremendous, believe me!"
            },
            'obama': {
                'engine': 'espeak',
                'voice': 'en+m7',           # Different male voice
                'speed': '140',             # Slower, thoughtful
                'pitch': '55',              # Higher pitch, smoother
                'amplitude': '180',
                'gap': '8',                 # More pauses
                'variant': 'm7',
                'prefix': "Uh, let me be clear, ",
                'suffix': " That's what we need to focus on."
            },
            'karen': {
                'engine': 'espeak',
                'voice': 'en+f4',           # Female voice
                'speed': '180',             # Fast, demanding
                'pitch': '80',              # High pitched, shrill
                'amplitude': '200',         # LOUD
                'gap': '2',                 # Rapid speech
                'variant': 'f4',
                'prefix': "Excuse me! ",
                'suffix': " I want to speak to your manager!"
            },
            'yoda': {
                'engine': 'espeak',
                'voice': 'en+m1',
                'speed': '120',             # Very slow
                'pitch': '30',              # Very deep
                'amplitude': '160',
                'gap': '15',                # Long pauses
                'variant': 'm1',
                'prefix': "Hmm. ",
                'suffix': " Strong with the force, you are."
            },
            'robot': {
                'engine': 'espeak',
                'voice': 'en+m2',
                'speed': '150',
                'pitch': '50',
                'amplitude': '200',
                'gap': '3',
                'variant': 'm2',
                'prefix': "BEEP BOOP. ",
                'suffix': " END TRANSMISSION."
            },
            'gangster': {
                'engine': 'espeak',
                'voice': 'en+m5',
                'speed': '130',             # Slower, menacing
                'pitch': '35',              # Deep, threatening
                'amplitude': '190',
                'gap': '6',
                'variant': 'm5',
                'prefix': "Listen here, see, ",
                'suffix': " Capisce?"
            },
            'surfer': {
                'engine': 'espeak',
                'voice': 'en+m6',
                'speed': '110',             # Very relaxed
                'pitch': '45',              # Laid back
                'amplitude': '150',         # Quieter
                'gap': '10',                # Lots of pauses, man
                'variant': 'm6',
                'prefix': "Dude, like, ",
                'suffix': " Totally gnarly, man!"
            },
            'anime': {
                'engine': 'espeak',
                'voice': 'en+f2',           # High female voice
                'speed': '200',             # Very fast
                'pitch': '90',              # Very high
                'amplitude': '180',
                'gap': '1',                 # Rapid cute speech
                'variant': 'f2',
                'prefix': "Kawaii! ",
                'suffix': " Desu ne~!"
            },
            'cowboy': {
                'engine': 'espeak',
                'voice': 'en+m4',
                'speed': '125',             # Drawling
                'pitch': '38',              # Deep western
                'amplitude': '170',
                'gap': '8',                 # Slow drawl
                'variant': 'm4',
                'prefix': "Well howdy partner, ",
                'suffix': " Yeehaw!"
            },
            'pirate': {
                'engine': 'espeak',
                'voice': 'en+m8',
                'speed': '140',
                'pitch': '42',              # Gruff
                'amplitude': '200',         # Loud and boisterous
                'gap': '5',
                'variant': 'm8',
                'prefix': "Arrr, matey! ",
                'suffix': " Shiver me timbers!"
            }
        }
        self.current_voice = 'trump'  # Default to Trump for maximum memes
        
    def speak_meme(self, text, voice_name='trump'):
        """Generate meme voice using eSpeak with character-specific parameters"""
        if voice_name not in self.voices:
            print(f"❌ Voice '{voice_name}' not available. Available: {list(self.voices.keys())}")
            voice_name = 'trump'  # Fallback to Trump
        
        voice_config = self.voices[voice_name]
        
        try:
            # Add character-specific prefix and suffix for more meme effect
            enhanced_text = voice_config['prefix'] + text + voice_config['suffix']
            
            # Build eSpeak command with voice parameters
            espeak_cmd = [
                'espeak',
                f"-v{voice_config['voice']}",          # Voice type
                f"-s{voice_config['speed']}",          # Speed (words per minute)
                f"-p{voice_config['pitch']}",          # Pitch (0-99)
                f"-a{voice_config['amplitude']}",      # Volume (0-200)
                f"-g{voice_config['gap']}",            # Gap between words
                enhanced_text
            ]
            
            print(f"🎤 {voice_name.upper()} says: '{text}'")
            print(f"🔧 Command: {' '.join(espeak_cmd)}")
            
            # Execute eSpeak command
            result = subprocess.run(espeak_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {voice_name.upper()} voice generation successful!")
                return True
            else:
                print(f"❌ eSpeak error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error generating {voice_name} voice: {e}")
            return False
    
    def speak_random_meme(self, text):
        """Speak with a random meme voice"""
        voice = random.choice(list(self.voices.keys()))
        print(f"🎲 Random voice selected: {voice}")
        return self.speak_meme(text, voice)
    
    def get_available_voices(self):
        """Get list of available meme voices"""
        return list(self.voices.keys())
    
    def set_voice(self, voice_name):
        """Set the default voice"""
        if voice_name in self.voices:
            self.current_voice = voice_name
            print(f"🎤 Default voice changed to: {voice_name.upper()}")
            return True
        else:
            print(f"❌ Voice '{voice_name}' not available")
            return False
    
    def test_voice(self, voice_name):
        """Test a specific voice with a sample message"""
        test_messages = {
            'trump': "hee yeahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh!",
            'obama': "This robot represents the kind of innovation that will move America forward into the future.",
            'karen': "This robot better follow that line perfectly or I'm calling corporate!",
            'yoda': "Follow the line, the robot must. Strong with the sensors, it is.",
            'robot': "Line detected. Following trajectory. Mission parameters nominal.",
            'gangster': "This robot's gonna follow that line, and it ain't gonna miss, see?",
            'surfer': "This robot is totally cruising along that line like it's riding the perfect wave.",
            'anime': "The robot-chan is so good at following lines! It's super kawaii!",
            'cowboy': "This here robot's gonna follow that line straighter than a sheriff's aim.",
            'pirate': "This robotic vessel be navigating the line better than any ship on the seven seas!"
        }
        
        message = test_messages.get(voice_name, "Testing voice output.")
        return self.speak_meme(message, voice_name)

def test_all_meme_voices():
    """Test all meme voices with character-appropriate messages"""
    print("🎭 TESTING ALL EPIC MEME VOICES!")
    print("=" * 60)
    
    meme_system = MemeVoiceSystem()
    
    for voice_name in meme_system.get_available_voices():
        print(f"\n🎵 Testing {voice_name.upper()} voice:")
        print("-" * 50)
        
        success = meme_system.test_voice(voice_name)
        if success:
            print(f"✅ {voice_name.upper()} voice test PASSED!")
        else:
            print(f"❌ {voice_name.upper()} voice test FAILED!")
        
        time.sleep(1)  # Brief pause between tests
    
    print("\n🎉 MEME VOICE TESTING COMPLETE!")
    print(f"🎤 Available voices: {', '.join(meme_system.get_available_voices())}")
    
    # Test random voice
    print("\n🎲 Testing RANDOM voice selection:")
    meme_system.speak_random_meme("Random voice test complete!")

def quick_trump_test():
    """Quick test of Trump voice specifically"""
    print("🇺🇸 QUICK TRUMP VOICE TEST!")
    meme_system = MemeVoiceSystem()
    
    trump_lines = [
        "he yeahhhhhh",
        "fuck yeahhhh!",
        "This robot has the best sensors, the best algorithms, tremendous technology!",
        "Nobody follows lines better than this robot, believe me!"
    ]
    
    for line in trump_lines:
        print(f"\n🎤 Trump says: '{line}'")
        meme_system.speak_meme(line, 'trump')
        time.sleep(1.5)

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Quick Trump test")
    print("2. Test all voices")
    print("3. Interactive mode")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        quick_trump_test()
    elif choice == "2":
        test_all_meme_voices()
    elif choice == "3":
        meme_system = MemeVoiceSystem()
        print("🎤 Interactive Meme Voice Mode!")
        print("Available voices:", ", ".join(meme_system.get_available_voices()))
        
        while True:
            voice = input("\nEnter voice name (or 'quit'): ").strip().lower()
            if voice == 'quit':
                break
            if voice in meme_system.get_available_voices():
                text = input("Enter text to speak: ").strip()
                if text:
                    meme_system.speak_meme(text, voice)
            else:
                print("Invalid voice. Try:", ", ".join(meme_system.get_available_voices()))
    else:
        quick_trump_test()  # Default to Trump test 