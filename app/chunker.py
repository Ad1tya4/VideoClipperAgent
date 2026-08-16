# app/chunker.py
"""
Cut the source video's audio into chunks, and record exactly where each chunk
sits in the video.

    We do not CALCULATE where chunks begin. We ask ffmpeg where it actually cut.

`-segment_time 45` is a request, not a guarantee. ffmpeg cuts at the next
packet boundary at or after 45 seconds, and it obviously cannot make the final
chunk 45 seconds long if there isnt that much of it.

Concretely, on a 290.6 second video the old arithmetic claimed the last chunk
covered 270 -> 315 seconds: 24 seconds of video that do not exist.

The fix is `-segment_list`, which makes ffmpeg write a manifest as it cuts:

    chunk_000.mp3,0.000000,45.000000
    chunk_001.mp3,45.000000,90.000000
    ...
    chunk_006.mp3,270.000000,290.690000

These are ground truth rather than inference. They matter for two reasons:

  * Every clip's cut point is computed as
        absolute_time = chunk_start_time + time_within_chunk
    so a wrong chunk start shifts every clip taken from that chunk.
  * The agent's decisions in M4 depend on how much of the video is covered by
    a usable transcript. Coverage computed from invented boundaries is a lie,
    and an agent reasoning from a lie makes confident wrong decisions.
"""

import csv
import subprocess
from pathlib import Path

from config import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    CHUNK_SECONDS,
    CHUNKS_DIR,
    SEGMENT_LIST_PATH,
    VIDEO_PATH,
    ensure_dirs,
)
from database import upsertSegmentBounds


def createAudioChunks(video_path: Path = VIDEO_PATH,
                      output_dir: Path = CHUNKS_DIR):
    """
    Strip the video track, downmix the audio, and split it into chunks.

    Returns the list of chunk boundaries ffmpeg reported.
    """
    ensure_dirs()

    if not video_path.exists():
        raise FileNotFoundError(
            f"No video at {video_path}. Put your file there, or change "
            f"VIDEO_PATH in config.py."
        )

    # Remove chunks from any previous run first. Without this, shortening the
    # video would leave the old, longer run's trailing chunks lying around and
    # they would be transcribed as if they belonged to the new video.
    for stale_chunk in output_dir.glob("chunk_*.mp3"):
        stale_chunk.unlink()

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",       # quiet unless something actually breaks
        "-y",                       # overwrite without prompting
        "-i", str(video_path),
        "-vn",                      # drop the video track, we only need audio
        "-ac", str(AUDIO_CHANNELS),     # mono
        "-ar", str(AUDIO_SAMPLE_RATE),  # 16 kHz - what Whisper uses internally
        "-acodec", "libmp3lame",
        "-f", "segment",
        "-segment_time", str(CHUNK_SECONDS),
        "-segment_list", str(SEGMENT_LIST_PATH),   # the manifest
        "-segment_list_type", "csv",
        "-reset_timestamps", "1",
        str(output_dir / "chunk_%03d.mp3"),
    ]

    # check=True turns a non-zero ffmpeg exit code into a Python exception,
    # rather than letting the pipeline carry on believing chunks exist.
    subprocess.run(command, check=True)

    return readChunkBoundaries()


def readChunkBoundaries(segment_list_path: Path = SEGMENT_LIST_PATH,
                        chunks_dir: Path = CHUNKS_DIR):
    """
    Parse ffmpeg's manifest into a list of dicts:

        {"chunk_name": "chunk_000.mp3", "path": Path(...),
         "start_time": 0.0, "end_time": 45.0}
    """
    if not segment_list_path.exists():
        raise FileNotFoundError(
            f"Expected ffmpeg's segment manifest at {segment_list_path} but it "
            f"is missing. Re-run chunking."
        )

    boundaries = []

    with open(segment_list_path, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 3:
                continue  # skip blank or malformed lines

            raw_name, start_text, end_text = row

            # ffmpeg writes only the basename here even when the output
            # pattern was an absolute path, but take .name anyway so this
            # does not depend on that behaviour.
            chunk_name = Path(raw_name).name

            boundaries.append(
                {
                    "chunk_name": chunk_name,
                    "path": chunks_dir / chunk_name,
                    "start_time": float(start_text),
                    "end_time": float(end_text),
                }
            )

    if not boundaries:
        raise ValueError(
            f"{segment_list_path} contained no usable rows - ffmpeg produced "
            f"no chunks. Is the video's audio track empty?"
        )

    return boundaries


def registerChunkBoundaries(boundaries: list[dict]):
    """
    Write the real boundaries into the database.

    This is deliberately separate from transcription. It only touches the
    timing columns, so chunks that are already transcribed keep their
    transcripts and their 'completed' status while still having their
    boundaries corrected.
    """
    for boundary in boundaries:
        upsertSegmentBounds(
            chunk_name=boundary["chunk_name"],
            start_time=boundary["start_time"],
            end_time=boundary["end_time"],
        )


def formatTime(seconds: float):
    """Seconds -> MM:SS, for anything a human is going to read."""
    total_seconds = int(seconds)
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def describeBoundaries(boundaries: list[dict]):
    """A short human-readable summary of what the chunker produced."""
    lines = []

    for boundary in boundaries:
        duration = boundary["end_time"] - boundary["start_time"]
        lines.append(
            f"  {boundary['chunk_name']}  "
            f"{formatTime(boundary['start_time'])} -> "
            f"{formatTime(boundary['end_time'])}  "
            f"({duration:.2f}s)"
        )

    total = boundaries[-1]["end_time"] if boundaries else 0.0
    lines.append(f"  total audio: {formatTime(total)} ({total:.2f}s)")

    return "\n".join(lines)
