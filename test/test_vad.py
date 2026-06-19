import unittest

import numpy as np

from voice_assistant.audio.filter import AudioFilter
from voice_assistant.config import AudioConfig


class FakeVad:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def is_speech(self, frame, sample_rate):
        return next(self.decisions, False)


class AudioFilterTests(unittest.TestCase):
    def test_emits_after_real_speech_then_silence(self):
        config = AudioConfig(
            pre_speech_ms=40,
            speech_start_ms=40,
            speech_end_ms=60,
            min_speech_ms=40,
        )
        decisions = [False, False, True, True, True, True, False, False, False]
        detector = AudioFilter(config, FakeVad(decisions))
        frame = np.full(config.frame_samples, 0.1, dtype=np.float32)

        results = [detector.process(frame) for _ in decisions]

        self.assertTrue(any(result.speech_started for result in results))
        utterances = [result.utterance for result in results if result.utterance is not None]
        self.assertEqual(len(utterances), 1)
        self.assertGreater(utterances[0].size, config.frame_samples * 4)

    def test_does_not_emit_short_noise(self):
        config = AudioConfig(
            pre_speech_ms=20,
            speech_start_ms=20,
            speech_end_ms=40,
            min_speech_ms=100,
        )
        decisions = [True, False, False]
        detector = AudioFilter(config, FakeVad(decisions))
        frame = np.full(config.frame_samples, 0.1, dtype=np.float32)

        results = [detector.process(frame) for _ in decisions]

        self.assertFalse(any(result.utterance is not None for result in results))
        self.assertTrue(results[-1].speech_ended)

    def test_counts_voiced_frames_from_entire_pre_roll(self):
        config = AudioConfig(
            pre_speech_ms=200,
            speech_start_ms=120,
            speech_end_ms=40,
            min_speech_ms=120,
        )
        decisions = [
            True, True, False, False, False, False,
            True, True, True, True,
            False, False,
        ]
        detector = AudioFilter(config, FakeVad(decisions))
        frame = np.full(config.frame_samples, 0.1, dtype=np.float32)

        results = [detector.process(frame) for _ in decisions]

        self.assertEqual(
            sum(result.speech_started for result in results),
            1,
        )
        self.assertEqual(
            sum(result.utterance is not None for result in results),
            1,
        )

    def test_rejects_wrong_frame_size_instead_of_silently_dropping_it(self):
        config = AudioConfig()
        detector = AudioFilter(config, FakeVad([]))

        with self.assertRaisesRegex(ValueError, "Expected 320 mono samples"):
            detector.process(np.zeros(100, dtype=np.float32))

    def test_validates_webrtc_configuration(self):
        with self.assertRaisesRegex(ValueError, "frame_ms must be 10, 20, or 30"):
            AudioFilter(AudioConfig(frame_ms=25), FakeVad([]))

        with self.assertRaisesRegex(ValueError, "sample_rate must be one of"):
            AudioFilter(AudioConfig(sample_rate=44100), FakeVad([]))

    def test_sanitizes_non_finite_device_samples(self):
        config = AudioConfig(
            pre_speech_ms=20,
            speech_start_ms=20,
            speech_end_ms=20,
            min_speech_ms=20,
        )
        detector = AudioFilter(config, FakeVad([True, False]))
        broken_frame = np.full(config.frame_samples, np.nan, dtype=np.float32)

        started = detector.process(broken_frame)
        ended = detector.process(broken_frame)

        self.assertTrue(started.speech_started)
        self.assertIsNotNone(ended.utterance)
        self.assertTrue(np.all(np.isfinite(ended.utterance)))


if __name__ == "__main__":
    unittest.main()
