# voice assistant

The audio pipeline listens continuously, uses WebRTC VAD to detect real speech
boundaries, transcribes when the user actually stops speaking, streams LLM
sentences to Kokoro, and supports barge-in while speech is playing.

Install dependencies with `uv sync`. Before running, place these model assets:

- A local faster-whisper model directory at `models/VoiceToText/Whisper`
- `kokoro-v1.0.onnx` at `models/TextToSpeech/kokoro/kokoro-v1.0.onnx`
- `voices-v1.0.bin` at `models/TextToSpeech/kokoro/voices-v1.0.bin`

Start the assistant with `uv run python detection/main/main.py`. Ollama must be
running with `llama3.2:1b` installed or you need to change it from `chatbot.py`(future updates will fix the error).

---


