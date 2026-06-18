import sounddevice as sd


class AudioCapture:
    """Capture microphone audio and push chunks into a queue."""

    def __init__(self, audio_queue, audio_config) -> None:
        self.audio_queue = audio_queue
        self.audio_config = audio_config

    def audio_callback(self, indata, frames, time, status) -> None:
        if status:
            print(f"Input stream status: {status}")
        self.audio_queue.put(indata.copy())

    def audio_stream_init(self) -> None:
        print("AudioCapture : audio_stream_init running...")
        with sd.InputStream(
            samplerate=self.audio_config.sample_rate,
            channels=1,
            dtype="float32",
            callback=self.audio_callback,
        ):
            input()
            self.audio_config.audio_stream_end = True

