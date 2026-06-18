import numpy as np


class AudioFilter:
    """Keep voiced chunks and mark a session complete after enough silence."""

    def __init__(self, audio_config) -> None:
        self.audio_config = audio_config

    def remove_silent_sound(self, chunk) -> None:
        rms = np.sqrt(np.mean(chunk**2))

        if rms > self.audio_config.silence_threshold:
            self.audio_config.buffer.append(chunk)
            self.audio_config.buffered_samples += 1
            self.audio_config.empty_audio_chunk = 0
            return

        if self.audio_config.buffered_samples == 0:
            return

        self.audio_config.empty_audio_chunk += 1
        if self.audio_config.empty_audio_chunk > self.audio_config.silence_chunks_to_end_session:
            self.audio_config.audio_session_end = True
            self.audio_config.empty_audio_chunk = 0

