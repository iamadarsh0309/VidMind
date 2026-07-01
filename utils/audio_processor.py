import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads/"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url : str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = { 
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
        "key": "FFmpegExtractAudio", 
        "preferredcodec": "wav", 
        "preferredquality": "192",
    }],
    "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav"   )
    return filename



def convert_to_wav(input_path:str) ->str:
    """convert any audio/video file to wav format using pydub"""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path:str, chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i,start in enumerate(range(0,len(audio), chunk_ms)):
        chunk = audio[start: start+chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def find_existing_chunks(wav_path: str) -> list[str] | None:
    """Return sorted chunk paths if they already exist for this wav."""
    import glob as glob_mod

    pattern = f"{wav_path}_chunk_*.wav"
    paths = sorted(glob_mod.glob(pattern), key=lambda p: int(p.rsplit("_chunk_", 1)[1].replace(".wav", "")))
    return paths if paths else None


def process_input(source: str, *, reuse_chunks: bool = False) -> list:
    if source.startswith("http") or source.startswith("https"):
        print("Downloading audio from YouTube...")
        wav_path = download_youtube_audio(source)
    elif source.lower().endswith(".wav"):
        wav_path = os.path.abspath(source)
        print(f"Using existing WAV: {wav_path}")
    else:
        print("Processing local audio file...")
        wav_path = convert_to_wav(source)

    if reuse_chunks:
        existing = find_existing_chunks(wav_path)
        if existing:
            print(f"Reusing {len(existing)} existing audio chunk(s) (no re-chunk).")
            return existing

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Generated {len(chunks)} chunks.")
    return chunks
