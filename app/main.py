# app/main.py
"""
Entry point.

Run it with:   python main.py       (from inside the app/ folder)
           or: python app/main.py   (from the project root)

Both now work identically, because every path is anchored in config.py rather
than being relative to the current working directory.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from chunker import (
    createAudioChunks,
    describeBoundaries,
    formatTime,
    registerChunkBoundaries,
)
from config import TRANSCRIBE_MODEL
from database import (
    getSegment,
    initialiseDatabase,
    saveCompletedSegment,
    saveFailedSegment,
)
from matcher import findMatch

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribeChunk(chunk_path):
    """
    Send one audio chunk to the speech-to-text model and return the text.

    # --- failure injection, temporary -------------------------------------
    # Uncomment to make one chunk fail so the resume path can be tested.
    # M6 replaces this with a proper --fault flag so the demo is reproducible
    # rather than depending on remembering to comment a line back out.
    #
    # if chunk_path.name == "chunk_002.mp3":
    #     raise TimeoutError("Simulated transcription timeout")
    # ----------------------------------------------------------------------
    """
    # Audio must be opened in binary mode - it is bytes, not text.
    with open(chunk_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=audio_file,
        )

    return transcription.text


def transcribeAllChunks(boundaries):
    """
    Walk every chunk, skipping any that a previous run already completed.

    This is the resume behaviour the brief asks for, and it is deliberately
    driven by what is IN THE DATABASE rather than by what is on disk. Chunk
    files are disposable and get recreated; the transcript is the expensive
    artefact and it is what we check before spending money again.
    """
    for boundary in boundaries:
        chunk_name = boundary["chunk_name"]
        chunk_path = boundary["path"]
        start_time = boundary["start_time"]
        end_time = boundary["end_time"]

        existing_segment = getSegment(chunk_name)

        if existing_segment is not None and existing_segment["status"] == "completed":
            print(f"[SKIP]    {chunk_name} already transcribed")
            continue

        if not chunk_path.exists():
            # The manifest listed a chunk that is not on disk. Record it as a
            # failure rather than crashing, so the rest of the video still
            # gets processed.
            saveFailedSegment(
                chunk_name=chunk_name,
                start_time=start_time,
                end_time=end_time,
                error=f"Chunk file missing at {chunk_path}",
            )
            print(f"[MISSING] {chunk_name} - no audio file on disk")
            continue

        print(
            f"\n[WORK]    {chunk_name} "
            f"[{formatTime(start_time)} - {formatTime(end_time)}]"
        )

        try:
            transcript = transcribeChunk(chunk_path)

            saveCompletedSegment(
                chunk_name=chunk_name,
                start_time=start_time,
                end_time=end_time,
                transcript=transcript,
            )

            print(f"[SUCCESS] {chunk_name}")
            print(f"          {transcript[:120]}...")

        # Catching broadly on purpose: any failure here must be RECORDED and
        # then survived, never allowed to kill the run and lose the chunks
        # already done. M5 replaces this with proper error classification so
        # a timeout and a bad API key are not treated identically.
        except Exception as error:
            saveFailedSegment(
                chunk_name=chunk_name,
                start_time=start_time,
                end_time=end_time,
                error=str(error),
            )

            print(f"[FAILED]  {chunk_name}")
            print(f"          reason: {error}")


def main():
    initialiseDatabase()

    # STEP 1 - cut the audio into chunks and learn their real boundaries.
    print("Chunking audio...")
    boundaries = createAudioChunks()
    registerChunkBoundaries(boundaries)
    print(describeBoundaries(boundaries))

    # STEP 2 - transcribe anything not already done.
    print("\nTranscribing...")
    transcribeAllChunks(boundaries)

    # STEP 3 - find the requested section.
    # Still a single interactive query. M3 replaces this with a batch of
    # requests loaded from a file, tracked in the database.
    query = input("\nWhat section would you like to find? ")

    verification, candidates = findMatch(query)

    print("\nLLM VERIFICATION")
    print(f"Matched:    {verification.matched}")
    print(f"Window:     {verification.window_id}")
    print(f"Confidence: {verification.confidence}")
    print(f"Reason:     {verification.reason}")


if __name__ == "__main__":
    main()
