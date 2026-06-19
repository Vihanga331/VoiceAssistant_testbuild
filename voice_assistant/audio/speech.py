import queue

import numpy as np
import sounddevice as sd


class VoiceToSpeech:
    """Generate and play interruptible TTS while the microphone stays live."""

    def __init__(self, voice_queue, audio_play_queue, audio_config) -> None:
        from kokoro_onnx import Kokoro
        self.voice_queue = voice_queue
        self.audio_play_queue = audio_play_queue
        self.config = audio_config
        self.kokoro = Kokoro(str(self.config.tts_model_path), str(self.config.tts_voices_path))

    def gen_voice(self) -> None:
        print("Speech synthesizer ready...")
        while not self.config.stop_event.is_set():
            try:
                turn_id, text = self.voice_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if turn_id != self.config.turn_id:
                continue
            if text is None:
                self.audio_play_queue.put((turn_id, None))
                continue
            try:
                audio, sample_rate = self.kokoro.create(
                    text, voice="af_sarah", speed=1.0, lang="en-us"
                )
                if turn_id == self.config.turn_id:
                    self.audio_play_queue.put((turn_id, (audio, sample_rate)))
            except Exception as error:
                print(f"Voice generation error: {error}")

    def audio_play(self) -> None:
        print("Audio player ready...")
        while not self.config.stop_event.is_set():
            try:
                turn_id, item = self.audio_play_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if turn_id != self.config.turn_id:
                continue
            if item is None:
                self.config.playback_active.clear()
                self.config.assistant_active.clear()
                self.config.interrupt_playback.clear()
                continue

            audio, sample_rate = item
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            block = max(1, int(sample_rate * 0.02))
            self.config.interrupt_playback.clear()
            self.config.playback_active.set()
            try:
                with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
                    for start in range(0, len(audio), block):
                        if (self.config.interrupt_playback.is_set()
                                or turn_id != self.config.turn_id
                                or self.config.stop_event.is_set()):
                            break
                        stream.write(audio[start : start + block].reshape(-1, 1))
            finally:
                self.config.playback_active.clear()
