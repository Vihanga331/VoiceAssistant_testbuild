from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faster_whisper import WhisperModel
from voice_assistant.app import Assistant
from voice_assistant.config import AudioConfig


SYSTEM_PROMPT = """
You are a helpful Assistant, give me response in few sentences.
Work with user collaborative.
-Please don't give response in markdown format.
-Don't include code snippets only provide text response.
"""



SYSTEM_PROMPT2 = """Act as a helful voice assistant.
do not include characters [*,-,]
"""

def build_assistant() -> Assistant:
    audio_config = AudioConfig()
    audio_config.validate_model_files()
    model = WhisperModel(
        str(audio_config.whisper_model_path),
        device="cpu",
        compute_type="int8",
        local_files_only=True,
    )

    return Assistant(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        audio_config=audio_config,
    )


if __name__ == "__main__":
    assistant = build_assistant()
    assistant.run()


# problem : Waiting 