import os
import subprocess

class SpeechEngine:
    """
    Speech Synthesis Engine: 
    Converts generated text into audible speech using system utilities.
    """
    def __init__(self):
        self.enabled = True
        # Check if 'espeak' or 'say' is available
        try:
            subprocess.run(["espeak", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.engine = "espeak"
        except:
            self.engine = None
            print("   [SPEECH] No TTS engine found (espeak). Speech will be simulated.")

    def speak(self, text):
        """Synthesizes speech from text."""
        if not self.enabled or not text:
            return
            
        print(f"   [SPEECH] Synthesizing: {text[:50]}...")
        
        if self.engine == "espeak":
            # Use espeak for Linux environments
            subprocess.run(["espeak", "-v", "en-us", text], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            # Simulation mode
            pass

    def toggle(self, status: bool):
        self.enabled = status
