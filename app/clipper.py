# app/clipper.py
"""
Cut a section out of the source video.

Four decisions are baked into this file, and each one has a wrong-looking
alternative that is faster:

1. RE-ENCODE INSTEAD OF STREAM-COPYING.
   `-c copy` is near-instant, but it can only cut at keyframes. Ask for a cut
   at 01:30 and you get one at whatever keyframe preceded it - often several
   seconds early, frequently opening on a frozen frame. For a tool whose only
   job is cutting at the right moment, that is the wrong thing to be fast at.
   Stream-copy is still available via CLIP_STREAM_COPY for when speed matters
   more than precision.

2. `-ss` BEFORE `-i`, AND `-t` RATHER THAN `-to`.
   Placing `-ss` before the input lets ffmpeg jump through the container
   rather than decoding from zero - fast, and still frame-accurate when
   re-encoding because it decodes from the preceding keyframe and throws the
   excess away. `-t` (a duration) is used instead of `-to` (an end time)
   because when `-ss` is an input option, what `-to` is measured against has
   varied between ffmpeg versions. A duration cannot be misread.

3. WRITE TO A TEMPORARY FILE, THEN RENAME.
   If ffmpeg is interrupted, a half-written clip must never be left sitting at
   the final path, because the next run would find a file that exists and
   assume the work was done. The rename is the last step, so a clip at its
   final path is always a finished clip.

4. VERIFY THE OUTPUT BEFORE CALLING IT A SUCCESS.
   ffmpeg can exit 0 and still produce something unusable. We probe the result
   and check its duration against what we asked for. "The command did not
   error" and "the clip is correct" are different claims, and the report in M6
   should only ever make the second one.
"""

import subprocess
from pathlib import Path

from config import (
    CLIP_DURATION_TOLERANCE,
    CLIP_PADDING_SECONDS,
    CLIP_STREAM_COPY,
    CLIPS_DIR,
    VIDEO_PATH,
    ensure_dirs,
)

# ffprobe is cheap but not free, and the source video's duration never changes
# during a run, so look it up once.
_duration_cache: dict[str, float] = {}


def probeDuration(media_path: Path):
    """
    Return a media file's duration in seconds, or None if it cannot be read.

    None is meaningful here: it is how a corrupt or truncated file announces
    itself. Callers treat it as a failure rather than as zero.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(media_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def getVideoDuration(video_path: Path = VIDEO_PATH):
    key = str(video_path)

    if key not in _duration_cache:
        duration = probeDuration(video_path)

        if duration is None:
            raise RuntimeError(f"Could not read the duration of {video_path}")

        _duration_cache[key] = duration

    return _duration_cache[key]


def buildClipPath(start_time: float, end_time: float,
                  clips_dir: Path = CLIPS_DIR):
    """
    Build a deterministic output path from the span alone.

    A clip's identity is WHAT IT CONTAINS, not the wording of the request that
    happened to produce it. Naming files after the query looks friendlier and
    is quietly wrong:

        "where does he speak about youtube"  -> window 0-90 -> file A
        "give me when youtube is spoken of"  -> window 0-90 -> file B

    Two names, one span, byte-identical video, and the reuse check never fires
    because it is looking for the wrong filename. Keying on the span instead
    makes both requests resolve to the same file, so the second one is
    recognised as already done.

    One decimal place, not whole seconds. Spans are chunk-aligned today so
    rounding is harmless, but M2 replaces window boundaries with utterance
    timestamps and spans become arbitrary floats. At that point 89.2-136.1 and
    89.4-135.8 would round to the same name, and the second request would be
    handed the first one's slightly-wrong clip while the duration check waved
    it through. 0.1s is finer than any difference that could matter.
    """
    # 06.1f -> "0089.0": four digits of seconds (up to 2h 46m) and one decimal.
    name = f"clip_{start_time:06.1f}-{end_time:06.1f}.mp4"
    return clips_dir / name


def verifyClip(clip_path: Path, expected_duration: float,
               tolerance: float = CLIP_DURATION_TOLERANCE):
    """
    Check that a clip on disk is actually usable.

    Returns (ok, reason). reason is None when ok is True.
    """
    if not clip_path.exists():
        return False, "file does not exist"

    if clip_path.stat().st_size == 0:
        return False, "file is empty"

    actual_duration = probeDuration(clip_path)

    if actual_duration is None:
        return False, "ffprobe could not read the file (likely truncated or corrupt)"

    difference = abs(actual_duration - expected_duration)

    if difference > tolerance:
        return False, (
            f"duration is {actual_duration:.2f}s but {expected_duration:.2f}s "
            f"was requested (off by {difference:.2f}s)"
        )

    return True, None


def buildFfmpegCommand(video_path: Path, output_path: Path,
                       start_time: float, duration: float):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start_time:.3f}",   # BEFORE -i: seek through the container
        "-i", str(video_path),
        "-t", f"{duration:.3f}",      # a duration, not an end time
    ]

    if CLIP_STREAM_COPY:
        # Fast, but cuts land on keyframes rather than where they were asked for.
        command += ["-c", "copy"]
    else:
        command += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "128k",
        ]

    command += [
        "-movflags", "+faststart",   # metadata at the front, so it streams
        str(output_path),
    ]

    return command


def cutClip(start_time: float, end_time: float,
            video_path: Path = VIDEO_PATH,
            padding: float = CLIP_PADDING_SECONDS):
    """
    Cut [start_time, end_time] (plus padding) out of the video.

    Deliberately knows nothing about clip requests, queries or matching. Give
    it the same video and the same span and it returns the same file, every
    time. That property is what makes the output content-addressable, and it
    is why the request text is not a parameter here - which request wanted
    this footage is the caller's business to record, not the cutter's.

    Returns a dict describing what happened:

        {"status": "created" | "reused" | "failed",
         "path": Path | None,
         "start": float, "end": float, "duration": float,
         "error": str | None}

    Nothing here raises on failure. A clip that cannot be cut is a result to
    be recorded and reported, not an exception that ends the run and discards
    every clip made before it.
    """
    ensure_dirs()

    video_duration = getVideoDuration(video_path)

    # Padding stops clips opening and closing mid-word. Clamping stops us
    # asking ffmpeg for footage that does not exist - which is exactly what
    # the old chunk arithmetic would have done at the end of the video.
    padded_start = max(0.0, start_time - padding)
    padded_end = min(video_duration, end_time + padding)
    duration = padded_end - padded_start

    if duration <= 0:
        return {
            "status": "failed",
            "path": None,
            "start": padded_start,
            "end": padded_end,
            "duration": 0.0,
            "error": (
                f"requested span {start_time:.2f}-{end_time:.2f}s is empty "
                f"after clamping to the video's {video_duration:.2f}s"
            ),
        }

    output_path = buildClipPath(padded_start, padded_end)

    # Already done on a previous run? Then it is completed work: leave it be.
    if output_path.exists():
        already_ok, _ = verifyClip(output_path, duration)

        if already_ok:
            return {
                "status": "reused",
                "path": output_path,
                "start": padded_start,
                "end": padded_end,
                "duration": duration,
                "error": None,
            }
        # Exists but is not valid - fall through and cut it again.

    # Write to a temporary name so an interrupted run cannot leave a partial
    # file sitting at the real path pretending to be finished.
    temporary_path = output_path.with_suffix(".partial.mp4")

    command = buildFfmpegCommand(video_path, temporary_path,
                                 padded_start, duration)

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        temporary_path.unlink(missing_ok=True)
        return {
            "status": "failed",
            "path": None,
            "start": padded_start,
            "end": padded_end,
            "duration": duration,
            "error": f"ffmpeg exited {result.returncode}: {result.stderr.strip()[:300]}",
        }

    ok, reason = verifyClip(temporary_path, duration)

    if not ok:
        temporary_path.unlink(missing_ok=True)
        return {
            "status": "failed",
            "path": None,
            "start": padded_start,
            "end": padded_end,
            "duration": duration,
            "error": f"clip failed verification: {reason}",
        }

    # Only now, with a verified file, does it get the real name.
    temporary_path.replace(output_path)

    return {
        "status": "created",
        "path": output_path,
        "start": padded_start,
        "end": padded_end,
        "duration": duration,
        "error": None,
    }
