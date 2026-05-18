from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.notes import create_notes


source = "https://www.youtube.com/watch?v=55pTFVoclvE"
language = "english"  # change to "english" to test Whisper


chunks = process_input(source)
transcript = transcribe_all(chunks, language=language)


print("\n=== TRANSCRIPT ===\n")
print(transcript)

print("\n=== TITLE ===\n")
title = generate_title(transcript)
print(title)

print("\n=== SUMMARY ===\n")
print(summarize(transcript))

print("\n=== NOTES ===\n")
create_notes(transcript, title)
