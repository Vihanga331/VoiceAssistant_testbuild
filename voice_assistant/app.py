import queue
import threading

from voice_assistant.audio.capture import AudioCapture
from voice_assistant.audio.filter import AudioFilter
from voice_assistant.audio.speech import VoiceToSpeech
from voice_assistant.audio.transcription import WhisperEngine
from voice_assistant.config import AudioConfig
from voice_assistant.llm.chatbot import Chatbot


class Assistant:
    """Full-duplex voice pipeline with VAD turn detection and barge-in."""

    def __init__(self, model, system_prompt, audio_config=None) -> None:
        self.audio_config = audio_config or AudioConfig()
        self.audio_queue = queue.Queue(maxsize=250)
        self.text_queue = queue.Queue()
        self.voice_queue = queue.Queue()
        self.audio_play_queue = queue.Queue()
        # self.llm_model = llm_model   # Edited

        self.audio_capture = AudioCapture(self.audio_queue, self.audio_config)
        audio_filter = AudioFilter(self.audio_config)
        self.whisper = WhisperEngine(
            model, self.audio_config, self.audio_queue, audio_filter, self.text_queue
        )
        self.chatbot = Chatbot(
            self.text_queue, self.voice_queue, self.audio_config, system_prompt,
        )
        self.voice_to_speech = VoiceToSpeech(
            self.voice_queue, self.audio_play_queue, self.audio_config
        )

    def run(self) -> None:
        workers = (
            self.audio_capture.audio_stream_init,
            self.whisper.vad_worker,
            self.whisper.transcription_worker,
            self.chatbot.ask_ai,
            self.voice_to_speech.gen_voice,
            self.voice_to_speech.audio_play,
        )
        threads = [threading.Thread(target=worker, daemon=True) for worker in workers]
        for thread in threads:
            thread.start()
        try:
            while all(thread.is_alive() for thread in threads):
                for thread in threads:
                    thread.join(timeout=0.25)
        except KeyboardInterrupt:
            print("\nStopping assistant...")
        finally:
            self.audio_config.stop_event.set()
            self.audio_config.request_barge_in()
            for thread in threads:
                thread.join(timeout=2)
