"""
Why this file exists:

if someone clones and runs the same script from two
different directories you silently get two different databases, with no
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
# exactly zero accuracy gain. Downmixing
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# --- models ----------------------------------------------------------------

TRANSCRIBE_MODEL = "whisper-1"
EMBEDDING_MODEL = "text-embedding-3-small"
VERIFICATION_MODEL = "gpt-5.6-terra"


def ensure_dirs():
    """Create the directories we write into. Safe to call as often as we like."""
    for directory in (CHUNKS_DIR, CLIPS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
