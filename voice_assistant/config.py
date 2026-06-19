from dataclasses import dataclass, field
from pathlib import Path
import threading


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AudioConfig:
    """Audio tuning and thread-safe runtime signals."""

    sample_rate: int = 16000
    frame_ms: int = 20
    vad_aggressiveness: int = 2
    pre_speech_ms: int = 300
    speech_start_ms: int = 120
    speech_end_ms: int = 650
    min_speech_ms: int = 250
    max_utterance_seconds: float = 30.0
    # While speakers are active, require this RMS to avoid treating our own TTS
    # as a barge-in. Headphones provide much better echo isolation.
    barge_in_rms: float = 0.035
    whisper_model_path: Path = PROJECT_ROOT / "models" / "VoiceToText" / "Whisper"
    tts_model_path: Path = PROJECT_ROOT / "models" / "TextToSpeech" / "kokoro" / "kokoro-v1.0.onnx"
    tts_voices_path: Path = PROJECT_ROOT / "models" / "TextToSpeech" / "kokoro" / "voices-v1.0.bin"
    transcription_path: Path = PROJECT_ROOT / "transcription.txt"

    stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    assistant_active: threading.Event = field(default_factory=threading.Event, init=False)
    playback_active: threading.Event = field(default_factory=threading.Event, init=False)
    interrupt_playback: threading.Event = field(default_factory=threading.Event, init=False)
    _turn_id: int = field(default=0, init=False)
    _turn_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    @property
    def turn_id(self) -> int:
        with self._turn_lock:
            return self._turn_id

    def begin_user_turn(self) -> int:
        """Invalidate outstanding LLM/TTS work and return the new turn id."""
        with self._turn_lock:
            self._turn_id += 1
            turn_id = self._turn_id
        self.interrupt_playback.set()
        return turn_id

    def request_barge_in(self) -> None:
        self.interrupt_playback.set()

    def validate_model_files(self) -> None:
        whisper_ready = all(
            (self.whisper_model_path / filename).is_file()
            for filename in ("config.json", "model.bin")
        )
        missing = [] if whisper_ready else [self.whisper_model_path]
        missing.extend(
            path for path in (self.tts_model_path, self.tts_voices_path)
            if not path.is_file()
        )
        if missing:
            paths = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Required local voice models are missing:\n"
                f"{paths}\nSee README.md for model setup."
            )
