from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    silence_threshold: float = 0.0001
    silence_chunks_to_end_session: int = 200
    max_buffered_chunks_before_transcribe: int = 200
    whisper_model_path: Path = PROJECT_ROOT / "models" / "VoiceToText" / "Whisper"
    tts_model_path: Path = PROJECT_ROOT / "models" / "TextToSpeech" / "kokoro" / "kokoro-v1.0.onnx"
    tts_voices_path: Path = PROJECT_ROOT / "models" / "TextToSpeech" / "kokoro" / "voices-v1.0.bin"
    transcription_path: Path = PROJECT_ROOT / "detection" / "main" / "transcription.txt"

    audio_stream_end: bool = False
    audio_session_end: bool = False
    latched_session: bool = False
    is_assistant_responding: bool = False
    buffered_samples: int = 0
    empty_audio_chunk: int = 0
    buffer: list = field(default_factory=list) 

    def start_assistant_response(self) -> None:
        self.is_assistant_responding = True
        self.audio_session_end = True
        self.latched_session = True

    def finish_assistant_response(self) -> None:
        self.is_assistant_responding = False
        self.audio_session_end = False
        self.latched_session = False
        self.empty_audio_chunk = 0
        self.clear_audio_buffer()

    def clear_audio_buffer(self) -> None:
        self.buffer.clear()
        self.buffered_samples = 0

