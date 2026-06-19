import queue


class WhisperEngine:
    """Segment live audio with VAD and transcribe complete utterances."""

    def __init__(self, model, audio_config, audio_queue, audio_filter, text_queue) -> None:
        self.model = model
        self.config = audio_config
        self.audio_queue = audio_queue
        self.audio_filter = audio_filter
        self.text_queue = text_queue
        self.utterance_queue = queue.Queue(maxsize=8)
        self._pending_turn_id = None

    def _transcribe(self, audio) -> str:
        segments, _ = self.model.transcribe(audio, vad_filter=False)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        if text:
            self.config.transcription_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.transcription_path.open("a", encoding="utf-8") as file:
                file.write(text + "\n")
            print(f"You: {text}")
        return text

    def vad_worker(self) -> None:
        """Keep VAD real-time even while Whisper is transcribing."""
        print("Voice activity detector ready...")
        while not self.config.stop_event.is_set():
            try:
                frame = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            result = self.audio_filter.process(frame)
            if result.speech_started and self.config.assistant_active.is_set():
                print("Barge-in detected; stopping assistant audio.")
                # Invalidate the old response immediately so already-generated
                # sentence audio cannot resume while this utterance finishes.
                self._pending_turn_id = self.config.begin_user_turn()
            if result.utterance is None:
                if result.speech_ended and self._pending_turn_id is not None:
                    # A noise burst can trigger start detection and still fail
                    # the minimum-speech check. Do not leak that barge-in state
                    # into the next real utterance.
                    self._pending_turn_id = None
                    self.config.assistant_active.clear()
                continue
            was_barge_in = self._pending_turn_id is not None
            turn_id = self._pending_turn_id or self.config.begin_user_turn()
            self._pending_turn_id = None
            try:
                self.utterance_queue.put_nowait(
                    (turn_id, was_barge_in, result.utterance)
                )
            except queue.Full:
                print("Transcription queue full; dropping oldest utterance.")
                self.utterance_queue.get_nowait()
                self.utterance_queue.put_nowait(
                    (turn_id, was_barge_in, result.utterance)
                )

    def transcription_worker(self) -> None:
        print("Speech recognizer ready...")
        while not self.config.stop_event.is_set():
            try:
                turn_id, was_barge_in, audio = self.utterance_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            text = self._transcribe(audio)
            if text:
                if turn_id == self.config.turn_id:
                    self.text_queue.put((turn_id, text))
            else:
                if was_barge_in and turn_id == self.config.turn_id:
                    self.config.assistant_active.clear()

    # Compatibility for callers of the original worker method.
    whisper_worker = transcription_worker
