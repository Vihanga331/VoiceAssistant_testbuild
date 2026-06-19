import queue

import sounddevice as sd


class AudioCapture:
    """Continuously capture exact WebRTC-VAD-sized microphone frames."""

    def __init__(self, audio_queue, audio_config) -> None:
        self.audio_queue = audio_queue
        self.audio_config = audio_config

    def audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"Input stream status: {status}")
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.audio_queue.put_nowait(indata.copy())

    def audio_stream_init(self) -> None:
        print("Microphone listening (Ctrl+C to stop)...")
        with sd.InputStream(
            samplerate=self.audio_config.sample_rate,
            blocksize=self.audio_config.frame_samples,
            channels=1,
            dtype="float32",
            latency="low",
            callback=self.audio_callback,
        ):
            self.audio_config.stop_event.wait()

