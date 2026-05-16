from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all


source = "http://youtube.com/watch?v=Y2Zq3wSMATw"
language = "english"  # change to "english" to test Whisper


chunks = process_input(source)
transcript = transcribe_all(chunks, language=language)


print("\n=== TRANSCRIPT ===\n")
print(transcript)
