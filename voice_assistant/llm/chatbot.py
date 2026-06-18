import time


class Chatbot:
    def __init__(self, text_queue, voice_queue, system_prompt, model_name="llama3.2:1b") -> None:
        self.text_queue = text_queue
        self.voice_queue = voice_queue
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.splitters = (".", "?", "!", "\n")

    def _flush_sentence_when_ready(self, buffer):
        if not buffer.strip():
            return ""

        if any(buffer.rstrip().endswith(splitter) for splitter in self.splitters):
            self.voice_queue.put(buffer.strip())
            return ""

        return buffer

    def ask_ai(self) -> None:
        print("Chatbot : ask_ai running...")
        time.sleep(2)

        while True:
            prompt = self.text_queue.get()
            print("LLM translating text")

            try:
                from ollama import chat

                stream = chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                )

                buffer = ""
                for chunk in stream:
                    part = chunk["message"]["content"]
                    print(part, end="", flush=True)
                    buffer += part
                    buffer = self._flush_sentence_when_ready(buffer)

                if buffer.strip():
                    self.voice_queue.put(buffer.strip())

                self.voice_queue.put(None)
                print()
            except KeyboardInterrupt:
                self.voice_queue.put(None)
            except Exception as error:
                print(f"Chatbot error: {error}")
                self.voice_queue.put(None)

