import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "data" / "testvid.MP4"
CHUNKS_DIR = BASE_DIR / "app" / "chunks"
# VIDEO_PATH = Path("data/testvid.MP4")
# CHUNKS_DIR = Path("chunks")
CHUNK_LENGTH = 45


def create_audio_chunks(video_path: Path, output_dir: Path):
    #remove vid, extract audio, chunkify
    # Create the chunks folder if it doesn't already exist

    output_dir.mkdir(exist_ok=True)

    output_pattern = output_dir / "chunk_%03d.mp3"
    # This list represents the ffmpeg terminal command
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-f", "segment",
        "-segment_time", str(CHUNK_LENGTH),
        "-reset_timestamps", "1",
        str(output_pattern),
    ]
    #acc executing the command
    subprocess.run(command, check=True)


def transcribe_chunk(chunk_path: Path):
    #send one audio chunk to OpenAI's speech-to-text model(whisper) via api
    #and returns the transcript as text.
    with open(chunk_path, "rb") as audio_file:
        # Audio files must be read as bytes, not normal text
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

    return transcription.text


def format_time(seconds: int):
    #Converts seconds into MM:SS
    minutes = seconds // 60
    remaining_seconds = seconds % 60

    return f"{minutes:02d}:{remaining_seconds:02d}"


def main():
    #STEP1
    #Turn our long video into lots of 45-second audio files
    create_audio_chunks(VIDEO_PATH, CHUNKS_DIR)
    #find every chunk in order from chunk folder
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.mp3"))
    #see where chunk starts ( 1 = 1* 45)
    for index, chunk_path in enumerate(chunk_files):
        start_time = index * CHUNK_LENGTH
        end_time = start_time + CHUNK_LENGTH

        print(
            f"\n[{format_time(start_time)} - {format_time(end_time)}]"
        )

        transcript = transcribe_chunk(chunk_path)

        print(transcript)


if __name__ == "__main__":
    main()