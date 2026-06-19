from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import webrtcvad


@dataclass(frozen=True)
class VADResult:
    """The state change produced by one VAD frame."""

    utterance: np.ndarray | None = None
    speech_started: bool = False
    speech_ended: bool = False


class AudioFilter:
    """Segment fixed-size microphone frames using WebRTC voice activity detection."""

    VALID_SAMPLE_RATES = {8000, 16000, 32000, 48000}
    VALID_FRAME_MS = {10, 20, 30}

    def __init__(self, audio_config, vad=None) -> None:
        self.config = audio_config
        self._validate_config()
        self.vad = vad or webrtcvad.Vad(audio_config.vad_aggressiveness)

        self._pre_roll_frames = max(
            1, math.ceil(audio_config.pre_speech_ms / audio_config.frame_ms)
        )
        self._start_frames = max(
            1, math.ceil(audio_config.speech_start_ms / audio_config.frame_ms)
        )
        self._end_frames = max(
            1, math.ceil(audio_config.speech_end_ms / audio_config.frame_ms)
        )
        self._minimum_voiced_frames = max(
            1, math.ceil(audio_config.min_speech_ms / audio_config.frame_ms)
        )
        self._maximum_frames = max(
            1,
            math.ceil(
                audio_config.max_utterance_seconds * 1000
                / audio_config.frame_ms
            ),
        )
        self._required_start_frames = max(1, math.ceil(self._start_frames * 2 / 3))

        # Keep each frame with its VAD decision. Counting only the shorter start
        # window loses the first part of short commands such as "stop" or "yes".
        self._pre_roll = deque(maxlen=self._pre_roll_frames)
        self._start_window = deque(maxlen=self._start_frames)
        self._frames: list[np.ndarray] = []
        self._speaking = False
        self._silence_frames = 0
        self._voiced_frames = 0

    def _validate_config(self) -> None:
        if (
            not isinstance(self.config.sample_rate, int)
            or self.config.sample_rate not in self.VALID_SAMPLE_RATES
        ):
            raise ValueError(
                "WebRTC VAD sample_rate must be one of "
                f"{sorted(self.VALID_SAMPLE_RATES)}, got {self.config.sample_rate}."
            )
        if (
            not isinstance(self.config.frame_ms, int)
            or self.config.frame_ms not in self.VALID_FRAME_MS
        ):
            raise ValueError(
                "WebRTC VAD frame_ms must be 10, 20, or 30, "
                f"got {self.config.frame_ms}."
            )
        if (
            not isinstance(self.config.vad_aggressiveness, int)
            or not 0 <= self.config.vad_aggressiveness <= 3
        ):
            raise ValueError("vad_aggressiveness must be between 0 and 3.")
        timings = (
            self.config.pre_speech_ms,
            self.config.speech_start_ms,
            self.config.speech_end_ms,
            self.config.min_speech_ms,
            self.config.max_utterance_seconds,
        )
        if not all(math.isfinite(value) and value > 0 for value in timings):
            raise ValueError("VAD timing settings must all be greater than zero.")
        maximum_ms = self.config.max_utterance_seconds * 1000
        if self.config.min_speech_ms > maximum_ms:
            raise ValueError("min_speech_ms cannot exceed max_utterance_seconds.")
        if self.config.speech_start_ms > maximum_ms:
            raise ValueError("speech_start_ms cannot exceed max_utterance_seconds.")
        if (
            not math.isfinite(self.config.barge_in_rms)
            or self.config.barge_in_rms < 0
        ):
            raise ValueError("barge_in_rms cannot be negative.")

    @staticmethod
    def _normalize_frame(frame: np.ndarray, expected_samples: int) -> np.ndarray:
        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        if samples.size != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} mono samples per VAD frame, "
                f"received {samples.size}."
            )
        # Device/driver glitches can produce NaN or infinity. They must never be
        # passed into the PCM conversion or counted as a barge-in.
        return np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

    @staticmethod
    def _pcm16(frame: np.ndarray) -> bytes:
        pcm = np.rint(np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2")
        return pcm.tobytes()

    def _reset_utterance(self) -> None:
        self._frames.clear()
        self._start_window.clear()
        self._pre_roll.clear()
        self._silence_frames = 0
        self._voiced_frames = 0
        self._speaking = False

    def _finish_utterance(self) -> VADResult:
        utterance = None
        if self._voiced_frames >= self._minimum_voiced_frames:
            utterance = np.concatenate(self._frames)
        self._reset_utterance()
        return VADResult(utterance=utterance, speech_ended=True)

    def process(self, frame: np.ndarray) -> VADResult:
        """Process exactly one frame and report speech start or utterance end."""
        frame = self._normalize_frame(frame, self.config.frame_samples)
        voiced = bool(
            self.vad.is_speech(self._pcm16(frame), self.config.sample_rate)
        )

        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        if self.config.playback_active.is_set() and rms < self.config.barge_in_rms:
            voiced = False

        if not self._speaking:
            saved_frame = frame.copy()
            self._pre_roll.append((saved_frame, voiced))
            self._start_window.append(voiced)

            if (
                len(self._start_window) < self._start_frames
                or sum(self._start_window) < self._required_start_frames
            ):
                return VADResult()

            self._speaking = True
            self._frames = [saved for saved, _ in self._pre_roll]
            self._voiced_frames = sum(
                was_voiced for _, was_voiced in self._pre_roll
            )
            return VADResult(speech_started=True)

        self._frames.append(frame.copy())
        if voiced:
            self._voiced_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        if (
            self._silence_frames >= self._end_frames
            or len(self._frames) >= self._maximum_frames
        ):
            return self._finish_utterance()
        return VADResult()
