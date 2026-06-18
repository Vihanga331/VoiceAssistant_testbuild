import numpy as np


class WhisperEngine:
    """Translate buffered microphone audio into text."""

    def __init__(self, model, audio_config, audio_queue, audio_filter, text_queue) -> None:
        self.model = model
        self.audio_config = audio_config
        self.audio_queue = audio_queue
        self.audio_filter = audio_filter
        self.text_queue = text_queue
        self.prompt_text = ""

    def translate_to_text(self) -> None:
        print("WhisperEngine : translate_to_text running...")
        if not self.audio_config.buffer:
            return

        audio = np.concatenate(self.audio_config.buffer, axis=0).flatten()
        self.audio_config.clear_audio_buffer()

        segments, info = self.model.transcribe(audio, vad_filter=True)
        with self.audio_config.transcription_path.open("a", encoding="utf-8") as file:
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                file.write(text + "\n")
                print(text)
                self.prompt_text += " " + text

    def _send_prompt_to_chatbot(self) -> None:
        prompt = self.prompt_text.strip()
        self.prompt_text = ""

        if not prompt:
            self.audio_config.finish_assistant_response()
            return

        self.audio_config.start_assistant_response()
        self.text_queue.put(prompt)

    def whisper_worker(self) -> None:
        print("WhisperEngine : whisper_worker running...")
        while True:
            chunk = self.audio_queue.get()

            if self.audio_config.audio_stream_end:
                print("Audio Stream ended")
                self.translate_to_text()
                self._send_prompt_to_chatbot()
                break

            if self.audio_config.is_assistant_responding:
                continue

            if not self.audio_config.audio_session_end:
                self.audio_filter.remove_silent_sound(chunk)

                if (
                    self.audio_config.buffered_samples
                    > self.audio_config.max_buffered_chunks_before_transcribe
                ):
                    print(
                        "Translating... : Buffered Samples : "
                        f"{self.audio_config.buffered_samples}"
                    )
                    self.translate_to_text()
                continue

            if not self.audio_config.latched_session:
                print("Audio Session ended")
                self.translate_to_text()
                self._send_prompt_to_chatbot()

