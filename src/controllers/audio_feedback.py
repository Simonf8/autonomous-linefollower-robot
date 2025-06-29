import subprocess
import threading
import queue
import shutil
import requests
import json
import tempfile
import os
import time
from typing import Optional, Dict, Any
import logging

class AudioFeedback:
    """Enhanced text-to-speech feedback system with multiple provider support."""

    def __init__(self, preferred_provider: str = "auto"):
        self.message_queue = queue.Queue()
        self.preferred_provider = preferred_provider
        self.available_providers = {}
        self.enabled = False
        
        # Initialize available providers
        self._check_providers()
        
        if self.enabled:
            # Start a worker thread to process the queue
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()
            print(f"Audio feedback system initialized with providers: {list(self.available_providers.keys())}")
        else:
            print("Audio feedback system disabled - no providers available.")

    def _check_providers(self):
        """Check which TTS providers are available."""
        
        # Check local pico2wave (original fallback)
        if self._check_pico2wave():
            self.available_providers['pico2wave'] = self._speak_pico2wave
            self.enabled = True
        
        # Check Edge TTS (Free Microsoft service)
        if self._check_edge_tts():
            self.available_providers['edge'] = self._speak_edge_tts
            self.enabled = True
        
        # Check gTTS (Free Google service)
        if self._check_gtts():
            self.available_providers['gtts'] = self._speak_gtts
            self.enabled = True
        
        # Check espeak (Lightweight fallback)
        if self._check_espeak():
            self.available_providers['espeak'] = self._speak_espeak
            self.enabled = True
        
        # Check festival (Alternative fallback)
        if self._check_festival():
            self.available_providers['festival'] = self._speak_festival
            self.enabled = True
        
        # Set preferred provider if auto-detection
        if self.preferred_provider == "auto":
            # Priority order: edge > gtts > pico2wave > espeak > festival
            if 'edge' in self.available_providers:
                self.preferred_provider = 'edge'
            elif 'gtts' in self.available_providers:
                self.preferred_provider = 'gtts'
            elif 'pico2wave' in self.available_providers:
                self.preferred_provider = 'pico2wave'
            elif 'espeak' in self.available_providers:
                self.preferred_provider = 'espeak'
            elif 'festival' in self.available_providers:
                self.preferred_provider = 'festival'

    def _check_pico2wave(self) -> bool:
        """Check if pico2wave and aplay are available."""
        return shutil.which("pico2wave") and shutil.which("aplay")

    def _check_edge_tts(self) -> bool:
        """Check if edge-tts is available (install with: pip install edge-tts)."""
        try:
            import edge_tts
            return shutil.which("aplay") or shutil.which("mpv") or shutil.which("ffplay")
        except ImportError:
            return False

    def _check_gtts(self) -> bool:
        """Check if gTTS is available (install with: pip install gtts)."""
        try:
            import gtts
            return shutil.which("aplay") or shutil.which("mpv") or shutil.which("ffplay")
        except ImportError:
            return False

    def _check_espeak(self) -> bool:
        """Check if espeak is available."""
        return shutil.which("espeak")

    def _check_festival(self) -> bool:
        """Check if festival is available."""
        return shutil.which("festival")

    def speak(self, text: str, force_provider: Optional[str] = None):
        """Add a message to the speech queue."""
        if not self.enabled:
            return
        
        # Don't queue up too many messages
        if self.message_queue.qsize() > 5:
            print("Audio queue full, dropping new message.")
            return

        provider = force_provider if force_provider in self.available_providers else self.preferred_provider
        self.message_queue.put((text, provider))

    def _process_queue(self):
        """Worker thread that consumes messages and speaks them."""
        while True:
            try:
                text_to_speak, provider = self.message_queue.get()
                
                # Try the preferred provider first, then fallback to others
                success = False
                providers_to_try = [provider] + [p for p in self.available_providers.keys() if p != provider]
                
                for provider_name in providers_to_try:
                    if provider_name in self.available_providers:
                        try:
                            self.available_providers[provider_name](text_to_speak)
                            success = True
                            break
                        except Exception as e:
                            print(f"TTS provider {provider_name} failed: {e}")
                            continue
                
                if not success:
                    print(f"All TTS providers failed for text: {text_to_speak}")
                    
            except Exception as e:
                print(f"Unexpected error in audio feedback thread: {e}")
            finally:
                self.message_queue.task_done()

    def _speak_pico2wave(self, text: str):
        """Speak using pico2wave (original method)."""
        wav_file = f"/tmp/feedback_pico_{threading.get_ident()}.wav"
        
        # Generate speech
        pico_cmd = ["pico2wave", "-w", wav_file, text]
        result = subprocess.run(pico_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise Exception(f"pico2wave failed: {result.stderr}")
        
        # Play speech
        aplay_cmd = ["aplay", "-q", wav_file]
        result = subprocess.run(aplay_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise Exception(f"aplay failed: {result.stderr}")
        
        # Cleanup
        try:
            os.remove(wav_file)
        except:
            pass

    def _speak_edge_tts(self, text: str):
        """Speak using Microsoft Edge TTS (free, high quality)."""
        import edge_tts
        import asyncio
        
        async def _generate_and_play():
            # Use a pleasant voice
            voice = "en-US-AriaNeural"  # Female, natural
            # voice = "en-US-GuyNeural"  # Male alternative
            
            wav_file = f"/tmp/feedback_edge_{threading.get_ident()}.wav"
            
            # Generate speech
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(wav_file)
            
            # Play speech
            player_cmd = None
            if shutil.which("aplay"):
                player_cmd = ["aplay", "-q", wav_file]
            elif shutil.which("mpv"):
                player_cmd = ["mpv", "--no-video", "--really-quiet", wav_file]
            elif shutil.which("ffplay"):
                player_cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_file]
            
            if player_cmd:
                result = subprocess.run(player_cmd, capture_output=True, timeout=15)
                if result.returncode != 0:
                    raise Exception(f"Audio player failed: {result.stderr}")
            
            # Cleanup
            try:
                os.remove(wav_file)
            except:
                pass
        
        # Run async function in thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_generate_and_play())
        finally:
            loop.close()

    def _speak_gtts(self, text: str):
        """Speak using Google Text-to-Speech (free, requires internet)."""
        from gtts import gTTS
        
        wav_file = f"/tmp/feedback_gtts_{threading.get_ident()}.mp3"
        
        # Generate speech
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(wav_file)
        
        # Play speech
        player_cmd = None
        if shutil.which("mpv"):
            player_cmd = ["mpv", "--no-video", "--really-quiet", wav_file]
        elif shutil.which("ffplay"):
            player_cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_file]
        elif shutil.which("aplay"):
            # Convert to wav first for aplay
            wav_converted = f"/tmp/feedback_gtts_converted_{threading.get_ident()}.wav"
            convert_cmd = ["ffmpeg", "-i", wav_file, "-y", wav_converted]
            subprocess.run(convert_cmd, capture_output=True, timeout=10)
            player_cmd = ["aplay", "-q", wav_converted]
            wav_file = wav_converted  # For cleanup
        
        if player_cmd:
            result = subprocess.run(player_cmd, capture_output=True, timeout=15)
            if result.returncode != 0:
                raise Exception(f"Audio player failed: {result.stderr}")
        
        # Cleanup
        try:
            os.remove(wav_file)
        except:
            pass

    def _speak_espeak(self, text: str):
        """Speak using espeak (lightweight, built-in)."""
        # espeak can output directly to audio
        espeak_cmd = ["espeak", "-s", "150", "-v", "en", text]
        result = subprocess.run(espeak_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise Exception(f"espeak failed: {result.stderr}")

    def _speak_festival(self, text: str):
        """Speak using festival TTS."""
        # Festival can read from stdin
        festival_cmd = ["festival", "--tts"]
        result = subprocess.run(festival_cmd, input=text, text=True, capture_output=True, timeout=10)
        if result.returncode != 0:
            raise Exception(f"festival failed: {result.stderr}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the audio feedback system."""
        return {
            "enabled": self.enabled,
            "preferred_provider": self.preferred_provider,
            "available_providers": list(self.available_providers.keys()),
            "queue_size": self.message_queue.qsize()
        }

    def set_provider(self, provider: str) -> bool:
        """Change the preferred TTS provider."""
        if provider in self.available_providers:
            self.preferred_provider = provider
            print(f"TTS provider changed to: {provider}")
            return True
        else:
            print(f"TTS provider {provider} not available. Available: {list(self.available_providers.keys())}")
            return False 