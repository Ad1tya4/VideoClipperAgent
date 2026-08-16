# app/config.py
"""
Every path and tunable the project uses, in one place.

Why this file exists:

Paths used to be written like `Path("state.db")`. That does NOT mean "the
database belonging to this project" - it means "state.db relative to whatever
directory you happened to launch python from". Run the same script from two
different directories and you silently get two different databases, with no
error to tell you. main.py already anchored its paths properly; database.py
did not. Two files, two different ideas of where the project lives.

Everything below is anchored to THIS FILE's location, so it cannot drift no
matter where python is invoked from.
"""

from pathlib import Path

# __file__ is  .../VideoClipperAgent/app/config.py
#   .parent   -> .../VideoClipperAgent/app
#   .parent   -> .../VideoClipperAgent
# .resolve() makes it absolute, which is what removes the dependency on the
# current working directory.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# --- inputs ----------------------------------------------------------------

VIDEO_PATH = PROJECT_ROOT / "data" / "testvid.MP4"

# The batch of clip requests to fulfil. See request_store.py for the format.
REQUESTS_PATH = PROJECT_ROOT / "requests.json"

# --- generated state (all gitignored, all rebuildable) ---------------------

DB_PATH = APP_DIR / "state.db"
CHUNKS_DIR = APP_DIR / "chunks"
CLIPS_DIR = APP_DIR / "clips"

# ffmpeg writes this manifest while it cuts, telling us exactly where each
# chunk starts and ends. See chunker.py for why we read it instead of doing
# the arithmetic ourselves.
SEGMENT_LIST_PATH = APP_DIR / "chunks.csv"

# --- audio chunking --------------------------------------------------------

CHUNK_SECONDS = 45

# Whisper resamples all input to 16 kHz mono before doing anything with it.
# Uploading 44.1 kHz stereo therefore costs upload time and bandwidth for
# exactly zero accuracy gain. Downmixing here makes every run noticeably
# faster, which matters when you are re-running the pipeline all day.
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# --- clipping --------------------------------------------------------------

# Seconds of lead-in and lead-out added around every clip, so clips do not
# open or close mid-word. Clamped to the real bounds of the video.
CLIP_PADDING_SECONDS = 1.0

# How far a finished clip's duration may differ from what was requested before
# we call it broken. Re-encoding shifts things by a frame or two; anything
# larger means something actually went wrong.
CLIP_DURATION_TOLERANCE = 1.0

# True = `-c copy`: near-instant, but cuts snap to keyframes and can land
# seconds away from the requested time. False = re-encode: slower, accurate.
CLIP_STREAM_COPY = False

# --- matching and the escalation ladder ------------------------------------

# How many transcript windows the LLM verifier sees at each stage.
# The baseline is deliberately narrow (cheap, precise); widening is a
# deliberate escalation, not the default.
TOP_K_BASELINE = 5
TOP_K_WIDE = 15

# How many alternative phrasings the expansion step generates.
QUERY_EXPANSIONS = 4

# A request gets at most this many strategies before a terminal decision.
# A budget stops one impossible request consuming the whole run's money, and
# makes "I stopped because I ran out of budget" a distinct, loggable reason
# from "I stopped because the evidence was conclusive".
MAX_STRATEGIES_PER_REQUEST = 4

# Below this LLM confidence a match is accepted but flagged for human review
# rather than reported as a clean hit.
LOW_CONFIDENCE_THRESHOLD = 0.55

# Transcript coverage at or above this counts as "the whole video is readable".
# Not 1.0, because summing float chunk durations will not land exactly.
FULL_COVERAGE_RATIO = 0.999

# How many times a single chunk may be re-transcribed by the recovery path.
MAX_RETRANSCRIBE_ATTEMPTS = 2

# --- models ----------------------------------------------------------------

TRANSCRIBE_MODEL = "whisper-1"
EMBEDDING_MODEL = "text-embedding-3-small"
VERIFICATION_MODEL = "gpt-5.6-terra"


def ensure_dirs():
    """Create the directories we write into. Safe to call as often as we like."""
    for directory in (CHUNKS_DIR, CLIPS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
