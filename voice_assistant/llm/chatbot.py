import queue


class Chatbot:
    """Stream LLM sentences to TTS, discarding superseded turns."""

    def __init__(self, text_queue, voice_queue, audio_config, system_prompt,
                 llm_model="llama3.2:1b") -> None:
        self.text_queue = text_queue
        self.voice_queue = voice_queue
        self.config = audio_config
        self.system_prompt = system_prompt
        self.model_name = llm_model    # Edited
        self.history = [{"role": "system", "content": system_prompt}]

    @staticmethod
    def _sentences(buffer: str):
        last = max(buffer.rfind(mark) for mark in (".", "?", "!", "\n"))
        if last < 0:
            return [], buffer
        complete = buffer[: last + 1].strip()
        return ([complete] if complete else []), buffer[last + 1 :]

    def ask_ai(self) -> None:
        print("LLM worker ready...")
        while not self.config.stop_event.is_set():
            try:
                turn_id, prompt = self.text_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if turn_id != self.config.turn_id:
                continue

            self.config.assistant_active.set()
            user_message = {"role": "user", "content": prompt}
            messages = self.history + [user_message]
            response = ""
            buffer = ""
            try:
                from ollama import chat
                stream = chat(model=self.model_name, messages=messages, stream=True)
                print("Assistant: ", end="", flush=True)
                for chunk in stream:
                    if turn_id != self.config.turn_id:
                        break
                    part = chunk["message"]["content"]
                    print(part, end="", flush=True)
                    response += part
                    buffer += part
                    sentences, buffer = self._sentences(buffer)
                    for sentence in sentences:
                        self.voice_queue.put((turn_id, sentence))
                if turn_id != self.config.turn_id:
                    continue
                if buffer.strip():
                    self.voice_queue.put((turn_id, buffer.strip()))
                self.voice_queue.put((turn_id, None))
                self.history.extend((user_message, {"role": "assistant", "content": response}))
                self.history = self.history[:1] + self.history[-20:]
                print()
            except Exception as error:
                print(f"\nChatbot error: {error}")
                self.voice_queue.put((turn_id, None))
