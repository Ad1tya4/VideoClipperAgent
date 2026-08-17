# app/main.py
"""
    python main.py --fault chunk_002.mp3    simulate that chunk failing to transcribe so we can demo recovery reporducibility

The pipeline:
    1. cut the video's audio into chunks, and learn their real boundaries
    2. transcribe any chunk not already transcribed ( we dont do extra work)
    3. register the batch of clip requests
    4. work each request that still needs work, escalating through strategies
    5. print what was clipped, what was not, and why
"""

import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

from agent import processRequest, shouldProcess
from chunker import (
    createAudioChunks,
    describeBoundaries,
    formatTime,
    registerChunkBoundaries,
)
from config import (
    CHUNKS_DIR,
    MAX_RETRANSCRIBE_ATTEMPTS,
    REQUESTS_PATH,
    TRANSCRIBE_MODEL,
)
from database import (
    getAllRequests,
    getCoverage,
    getDecisionLog,
    getSegment,
    initialiseDatabase,
    logDecision,
    saveCompletedSegment,
    saveFailedSegment,
    saveUtterances,
    transcriptVersion,
)
from matcher import ensureSearchIndex
from request_store import syncRequests, writeExampleRequests

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chunks that should pretend to fail this run. Set from --fault.
#
# Failure injection is a command-line flag rather than a commented-out line of
# code because a demo that depends on remembering to comment something back in
# is a demo that will misfire on camera. This way run 1 and run 2 differ by one
# visible argument, and anyone cloning the repo can reproduce the failure.
FAULT_CHUNKS: set[str] = set()


def transcribeChunk(chunk_path, prompt: str = None, temperature: float = 0.0):
    """
    Send one audio chunk to the speech-to-text model.

    `prompt` and `temperature` exist so the recovery path can retry a failed
    chunk with DIFFERENT PARAMETERS rather than doing the same thing again and
    hoping. Whisper takes a prompt as a vocabulary hint, so seeding it with the
    clip request biases decoding toward the words being looked for - which is
    exactly the situation where a chunk is worth re-reading.
    """
    if chunk_path.name in FAULT_CHUNKS:
        raise TimeoutError(
            f"Simulated transcription timeout for {chunk_path.name} (--fault)"
        )

    with open(chunk_path, "rb") as audio_file:
        kwargs = {
            "model": TRANSCRIBE_MODEL,
            "file": audio_file,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }

        if prompt:
            kwargs["prompt"] = prompt[:220]
            kwargs["temperature"] = temperature

        transcription = client.audio.transcriptions.create(**kwargs)

    return transcription.text, getattr(transcription, "segments", None)


def transcribeAllChunks(boundaries):
    """
    Walk every chunk, skipping any a previous run already completed.

    Driven by what is IN THE DATABASE rather than what is on disk: chunk files
    are disposable and get recreated every run, while the transcript is the
    expensive artefact and is what we check before spending money again.
    """
    for boundary in boundaries:
        chunk_name = boundary["chunk_name"]
        chunk_path = boundary["path"]
        start_time = boundary["start_time"]
        end_time = boundary["end_time"]

        existing = getSegment(chunk_name)

        if existing is not None and existing["status"] == "completed":
            print(f"[SKIP]    {chunk_name} already transcribed")
            continue

        if not chunk_path.exists():
            saveFailedSegment(chunk_name, start_time, end_time,
                              f"Chunk file missing at {chunk_path}")
            print(f"[MISSING] {chunk_name}")
            continue

        print(f"[WORK]    {chunk_name} "
              f"[{formatTime(start_time)} - {formatTime(end_time)}]")

        try:
            transcript, utterances = transcribeChunk(chunk_path)
            saveCompletedSegment(chunk_name, start_time, end_time, transcript)
            saveUtterances(chunk_name, start_time, utterances)
            print(f"[SUCCESS] {chunk_name} "
                  f"({len(utterances or [])} utterances)")

        # Broad on purpose: any failure here must be RECORDED and survived,
        # never allowed to end the run and discard the chunks already done.
        except Exception as error:
            saveFailedSegment(chunk_name, start_time, end_time, str(error))
            print(f"[FAILED]  {chunk_name} - {error}")


def retranscribeSegment(segment, query: str):
    """
    recovery(  agent's `retranscribe` strat)

    Called only when a request could not be matched AND the transcript has
    holes - i.e. when the missing content might genuinely be in the audio we
    failed to read. Retries that ONE chunk with different parameters, leaving
    every other transcript untouched.
    """
    chunk_name = segment["chunk_name"]
    attempts_so_far = segment["retry_count"] or 0

    if attempts_so_far > MAX_RETRANSCRIBE_ATTEMPTS:
        logDecision(
            stage="transcribe", chunk_name=chunk_name,
            attempt=attempts_so_far, strategy="retranscribe",
            signals={"retry_count": attempts_so_far,
                     "max": MAX_RETRANSCRIBE_ATTEMPTS},
            decision="give_up",
            reasoning=(
                f"{chunk_name} has already failed {attempts_so_far} times. "
                f"Further identical retries are unlikely to differ; leaving it "
                f"as a permanent gap and reporting the gap honestly."
            ),
        )
        return False

    chunk_path = CHUNKS_DIR / chunk_name

    if not chunk_path.exists():
        return False

    print(f"[RETRY]   re-transcribing {chunk_name} with a vocabulary hint")

    try:
        transcript, utterances = transcribeChunk(
            chunk_path, prompt=query, temperature=0.2
        )

        saveCompletedSegment(chunk_name, segment["start_time"],
                             segment["end_time"], transcript)
        saveUtterances(chunk_name, segment["start_time"], utterances)

        logDecision(
            stage="transcribe", chunk_name=chunk_name,
            attempt=attempts_so_far + 1, strategy="retranscribe",
            signals={"retry_count": attempts_so_far,
                     "prompt_hint": query[:80], "temperature": 0.2},
            decision="retry_succeeded",
            reasoning=(
                f"Re-transcribed {chunk_name} with the clip request as a "
                f"vocabulary hint. The gap is closed and the search index will "
                f"be rebuilt to include it."
            ),
        )

        print(f"[RECOVER] {chunk_name} transcribed on retry")
        return True

    except Exception as error:
        saveFailedSegment(chunk_name, segment["start_time"],
                          segment["end_time"], str(error))

        logDecision(
            stage="transcribe", chunk_name=chunk_name,
            attempt=attempts_so_far + 1, strategy="retranscribe",
            signals={"retry_count": attempts_so_far, "error": str(error)[:200]},
            decision="retry_failed",
            reasoning=(
                f"Re-transcribing {chunk_name} with different parameters "
                f"failed again: {error}. This region stays a gap."
            ),
        )

        print(f"[FAILED]  {chunk_name} on retry - {error}")
        return False


def fulfilRequests(active_ids):
    version = transcriptVersion()
    print(f"\nTranscript version: {version}")

    for request in getAllRequests(active_ids):
        label = request["label"] or request["request_id"]
        process, reason = shouldProcess(request, version)

        if not process:
            print(f"\n[SKIP]    {label}: {reason}")
            continue

        print(f"\n[WORK]    {label}: {reason}")
        print(f'          "{request["query"]}"')

        outcome = processRequest(request, retranscribe_fn=retranscribeSegment)

        print(f"[{outcome['status'].upper()}] {label} "
              f"(strategies: {', '.join(outcome['strategies']) or 'none'})")


def printSummary(active_ids):
    """
    The final report: what was clipped, what was not, and why.

    The 'why' is the part usually missing, and it is the part the brief asks
    for explicitly.
    """
    coverage = getCoverage()
    requests = getAllRequests(active_ids)

    print("\n" + "=" * 76)
    print("RUN SUMMARY")
    print("=" * 76)

    print(f"\nTranscript coverage: {coverage['covered']:.1f}s of "
          f"{coverage['total']:.1f}s ({coverage['ratio'] * 100:.1f}%)")

    if coverage["gaps"]:
        for gap in coverage["gaps"]:
            print(f"  GAP  {formatTime(gap['start'])}-{formatTime(gap['end'])}"
                  f"  ({gap['chunk_name']})")
    else:
        print("  No gaps - the whole recording is searchable.")

    counts = {}
    for request in requests:
        counts[request["status"]] = counts.get(request["status"], 0) + 1

    print("\nRequests: " + ", ".join(
        f"{n} {status}" for status, n in sorted(counts.items())))

    for request in requests:
        label = request["label"] or request["request_id"]

        print("\n" + "-" * 76)
        print(f"{label}  [{request['status'].upper()}]")
        print(f'  asked:   "{request["query"]}"')

        if request["clip_path"]:
            print(f"  clip:    {os.path.basename(request['clip_path'])}  "
                  f"({request['start_time']:.1f}-{request['end_time']:.1f}s)")

        if request["resolution"]:
            print(f"  why:     {request['resolution']}")

        entries = getDecisionLog(request["request_id"])

        if entries:
            print("  decisions:")
            for entry in entries:
                print(f"    {entry['strategy']:>12} -> {entry['decision']}")

    print("\n" + "=" * 76)
    print("Full reasoning for every decision is in the decision_log table.")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Video clipper agent")
    parser.add_argument(
        "--fault", action="append", default=[], metavar="CHUNK",
        help="simulate a transcription failure for this chunk "
             "(e.g. --fault chunk_002.mp3). Repeatable.",
    )
    args = parser.parse_args()

    FAULT_CHUNKS.update(args.fault)

    if FAULT_CHUNKS:
        print(f"!! FAULT INJECTION ACTIVE: {', '.join(sorted(FAULT_CHUNKS))}\n")

    initialiseDatabase()

    print("Chunking audio...")
    boundaries = createAudioChunks()
    registerChunkBoundaries(boundaries)
    print(describeBoundaries(boundaries))

    print("\nTranscribing...")
    transcribeAllChunks(boundaries)

    print("\nBuilding search index...")
    rebuilt = ensureSearchIndex()
    print("  rebuilt from the current transcript"
          if rebuilt else "  unchanged since last run - reused")

    if not REQUESTS_PATH.exists():
        writeExampleRequests()
        print(f"\nCreated a starter requests file at {REQUESTS_PATH}")

    active_ids = syncRequests()

    fulfilRequests(active_ids)
    printSummary(active_ids)


if __name__ == "__main__":
    main()
