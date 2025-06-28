import subprocess
import threading
import queue
import shutil

class AudioFeedback:
    """Handles text-to-speech feedback in a non-blocking way."""

    def __init__(self):
        self.message_queue = queue.Queue()
        self.enabled = self._check_dependencies()
        
        if self.enabled:
            # Start a worker thread to process the queue
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()
            print("Audio feedback system initialized.")
        else:
            print("Audio feedback system disabled due to missing dependencies.")


    def _check_dependencies(self) -> bool:
        """Check if TTS and audio playback tools are installed."""
        if not shutil.which("pico2wave"):
            print("="*50)
            print("WARNING: 'pico2wave' not found.")
            print("Audio feedback will be disabled.")
            print("Please install it with: sudo apt-get update && sudo apt-get install libttspico-utils")
            print("="*50)
            return False
        if not shutil.which("aplay"):
            print("="*50)
            print("WARNING: 'aplay' not found.")
            print("Audio feedback will be disabled.")
            print("Please install it with: sudo apt-get update && sudo apt-get install alsa-utils")
            print("="*50)
            return False
        return True

    def speak(self, text: str):
        """
        Add a message to the speech queue to be spoken.
        """
        if not self.enabled:
            return
        
        # Don't queue up too many messages to prevent lag
        if self.message_queue.qsize() > 5:
            print("Audio queue full, dropping new message.")
            return

        self.message_queue.put(text)

    def _process_queue(self):
        """Worker thread that consumes messages and speaks them."""
        while True:
            try:
                text_to_speak = self.message_queue.get()
                
                # Use a unique temp file to avoid potential race conditions
                wav_file = f"/tmp/feedback_{threading.get_ident()}.wav"
                
                # Command to generate the WAV file
                pico_cmd = ["pico2wave", "-w", wav_file, text_to_speak]
                
                # Command to play the WAV file (-q for quiet mode)
                aplay_cmd = ["aplay", "-q", wav_file]

                # Generate the speech file
                result = subprocess.run(pico_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error generating speech with pico2wave: {result.stderr}")
                    continue

                # Play the speech file
                result = subprocess.run(aplay_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error playing speech with aplay: {result.stderr}")

            except Exception as e:
                print(f"An unexpected error occurred in the audio feedback thread: {e}")
            finally:
                # Ensure task_done is called even if errors occur
                if 'text_to_speak' in locals():
                    self.message_queue.task_done() 