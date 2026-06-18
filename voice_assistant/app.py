import queue
import threading

from voice_assistant.audio.capture import AudioCapture
from voice_assistant.audio.filter import AudioFilter
from voice_assistant.audio.speech import VoiceToSpeech
from voice_assistant.audio.transcription import WhisperEngine
from voice_assistant.config import AudioConfig
from voice_assistant.llm.chatbot import Chatbot


class Assistant:
    def __init__(self, model, system_prompt, audio_config=None) -> None:
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.voice_queue = queue.Queue()
        self.audio_play_queue = queue.Queue()

        self.audio_config = audio_config or AudioConfig()
        self.audio_capture = AudioCapture(self.audio_queue, self.audio_config)
        self.audio_filter = AudioFilter(self.audio_config)
        self.whisper = WhisperEngine(
            model=model,
            audio_config=self.audio_config,
            audio_queue=self.audio_queue,
            audio_filter=self.audio_filter,
            text_queue=self.text_queue,
        )
        self.chatbot = Chatbot(
            text_queue=self.text_queue,
            voice_queue=self.voice_queue,
            system_prompt=system_prompt,
        )
        self.voice_to_speech = VoiceToSpeech(
            voice_queue=self.voice_queue,
            audio_play_queue=self.audio_play_queue,
            audio_config=self.audio_config,
        )

    def run(self) -> None:
        threads = [
            threading.Thread(target=self.audio_capture.audio_stream_init, daemon=True),
            threading.Thread(target=self.whisper.whisper_worker, daemon=True),
            threading.Thread(target=self.chatbot.ask_ai, daemon=True),
            threading.Thread(target=self.voice_to_speech.gen_voice, daemon=True),
            threading.Thread(target=self.voice_to_speech.audio_play, daemon=True),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

