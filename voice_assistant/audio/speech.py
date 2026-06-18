import sounddevice as sd


class VoiceToSpeech:
    """Generate TTS audio and play it before listening resumes."""

    def __init__(self, voice_queue, audio_play_queue, audio_config) -> None:
        from kokoro_onnx import Kokoro

        self.voice_queue = voice_queue
        self.audio_play_queue = audio_play_queue
        self.audio_config = audio_config
        self.kokoro = Kokoro(
            str(self.audio_config.tts_model_path),
            str(self.audio_config.tts_voices_path),
        )

    def gen_voice(self) -> None:
        print("VoiceToSpeech : gen_voice running...")

        while True:
            text = self.voice_queue.get()

            if text is None:
                self.audio_play_queue.put(None)
                continue

            print("\n" + "=" * 50)

            try:
                audio, sample_rate = self.kokoro.create(
                    text,
                    voice="af_sarah",
                    speed=1.0,
                    lang="en-us",
                )
            except Exception as error:
                print(f"Voice generation error: {error}")
                continue

            self.audio_play_queue.put((audio, sample_rate))

    def audio_play(self) -> None:
        print("VoiceToSpeech : audio_play running...")

        while True:
            item = self.audio_play_queue.get()

            if item is None:
                self.audio_config.finish_assistant_response()
                print("Assistant audio finished. Listening again.")
                continue

            audio, sample_rate = item
            sd.play(audio, sample_rate)
            sd.wait()

