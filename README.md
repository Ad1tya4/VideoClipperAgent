# Video Clipper Agent

---

## What it does

- Splits a video's audio into chunks and transcribes them, **resuming from wherever it stopped** if a run is interrupted or a chunk fails
- Searches the transcript for each clip request using embeddings, verified by an LLM
- When a request doesn't match, **escalates through four strategies** before giving a verdict
- Distinguishes **"it isn't there"** from **"I couldn't read part of the video, so I don't know"**. These are different claims and are reported differently
- Cuts frame-accurate clips at sentence boundaries, not chunk boundaries
- **Never repeats expensive work**: transcripts, embeddings and finished clips all survive across runs
- Logs every decision with the numbers behind it

---

## Quick start

### Prerequisites

**Python 3.10+** and **ffmpeg** (which includes `ffprobe`). ffmpeg is not a pip package:

| Platform | Install |
|---|---|
| Windows | `winget install ffmpeg`  or [download](https://ffmpeg.org/download.html) and add to PATH |
| macOS | `brew install ffmpeg` |
| Debian/Ubuntu | `sudo apt install ffmpeg` |

Check it worked:

```bash
ffmpeg -version
ffprobe -version
```

### Setup

```bash
git clone https://github.com/Ad1tya4/VideoClipperAgent.git
cd VideoClipperAgent

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

Add your API key:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

then edit `.env` and set `OPENAI_API_KEY`.

### Add a video

Drop any video into `data/`. **That's it**  the agent will find it automatically, whatever it's called:

```
data/
  my_recording.mp4
```

Supported: `.mp4 .mov .mkv .webm .avi .m4v .mpg .mpeg .wmv .flv`

If you keep more than one video in `data/`, the agent stops and names them rather than guessing which you meant. Pick one explicitly in `.env`:

```
VIDEO_PATH=data/my_recording.mp4
```

Use a video **at least 2 minutes long** with clear speech on distinguishable topics, or there's nothing for the requests to discriminate between.

### Write your clip requests

Edit `requests.json` at the project root:

```json
[
  { "label": "budget",  "query": "the part where they discuss budget constraints" },
  { "label": "q3",      "query": "the section about Q3 results" }
]
```
Note: my requests.json has id instead of label both are purely cosmetic
### Run it

```bash
cd app
python main.py
```

Clips land in `app/clips/`. All commands run from inside `app/`.

---

## Seeing the failure recovery

The whole point of the project is what happens when transcription fails, so failure is injectable with a flag rather than hidden behind a commented-out line. These runs are show in the demo video.

```bash
cd app

python main.py --fault chunk_002.mp3     # run 1: that chunk fails
python main.py                           # run 2: recovers
```
---



## How it fits together

```
app/
  main.py           entry point, transcription loop, report, --fault injection
  config.py         all paths and tunables; video discovery
  chunker.py        ffmpeg audio splitting + real boundary detection
  database.py       every read and write to SQLite
  matcher.py        embeddings, retrieval strategies, LLM verification, narrowing
  agent.py          the decision layer: shouldProcess, route, the ladder
  clipper.py        ffmpeg cutting, verification, idempotent naming
  request_store.py  loading and validating requests.json

data/               your video goes here
requests.json       your clip requests
app/clips/          output
app/state.db        all persistent state
```

Pipeline order: **chunk → transcribe → index → fulfil each request → report.**



## What I'd do next with more time

Things I found, but didn't have time to fix before the deadline:

- **No input fingerprinting.** If the video is replaced but state.db is kept, the program could reuse transcripts from the old video. I would store a fingerprint of the input video and chunking settings, and if it changes, rebuild the relevant pipeline state.
- **Errors aren't classified.** The program treats temporary errors, like a timeout, the same as permanent errors, like an invalid API key. I would classify errors so temporary ones can be retried, while permanent ones stop the run immediately instead of wasting retries.

- **Thresholds are hand-tuned.** Values like `0.55` confidence, `8s` minimum clip length, and `0.999` coverage were chosen manually from one demo video. I would test them on labelled examples across multiple videos and tune them based on measured results.

- **Lexical fallback is exact-match only.** The main semantic + LLM matching can handle paraphrases and wording differences, but the lexical recovery pass still depends on exact word matches. I would add fuzzy matching to make that fallback more robust to transcription errors, especially for names and proper nouns.
- **Search windows are fixed in size.** Each search window covers two chunks, so longer discussions may be split across multiple windows. I would use variable or hierarchical windows so both short mentions and longer discussions can be matched properly.

- **Only one clip is returned per request.** If the same topic appears in several places, the system only returns the best match. I would allow it to return every strong matching region when the request calls for multiple occurrences.

- **Transcription is sequential.** Chunks are transcribed one after another even though they are independent. I would transcribe multiple chunks in parallel to reduce runtime on longer videos.

- **There are no automated tests.** Important logic such as `route()` is designed as a pure function, which makes it easy to test, but those tests have not been added yet. I would add unit tests for every decision path so changes do not silently break the agent's behaviour.

<p>
These limitations also highlight the need for stronger testing and evaluation, which is an essential part of developing a reliable system. The next step would be to test the system on longer, noisier videos with labelled ground-truth answers and measure things like retrieval accuracy, false positives, recovery success, latency, and cost.
</p>

## Engineering decisions

The implementation is designed around one principle: **completed work should never be discarded or unnecessarily repeated**. Expensive artefacts such as transcripts, embeddings and clips are persisted, while cheap deterministic artefacts such as audio chunks and search windows can be recreated when needed.

### Persistent, resumable state

Each audio chunk is tracked in SQLite with its boundaries, transcription status, transcript, error and retry count.

Before transcribing a chunk, the system checks the database:

- completed chunks are skipped
- failed chunks can be retried independently
- successful work from earlier runs is preserved

Database writes are transactional, so a state change either commits completely or rolls back. This prevents partially-written state from being treated as completed work.

### Recovery decisions

A request that cannot be matched does not immediately fail. It moves through a small escalation ladder:

| Strategy | Purpose |
|---|---|
| **baseline** | Retrieve the strongest transcript windows using embedding similarity and verify them with an LLM |
| **expand** | Rewrite the request to handle vocabulary and phrasing differences |
| **widen** | Search more candidates and add lexical matching for names, acronyms and numbers |
| **retranscribe** | Retry only failed chunks when missing transcript evidence could contain the answer |

Re-transcription is only attempted when transcript coverage contains gaps. If the entire video has already been successfully transcribed, another transcription call cannot reveal new evidence and is therefore skipped.

### Deterministic control flow

The LLM is used for semantic judgement:

> Does this transcript region mean what the user asked for?

The agent's control flow is handled by deterministic Python:

> Should this result be accepted, held, retried, skipped or escalated?

Keeping these responsibilities separate makes the workflow reproducible, testable and easier to audit.

### `UNFOUND` vs `ESCALATED`

A failed match can mean two very different things:

| Result | Meaning |
|---|---|
| **UNFOUND** | The whole video was searchable, all strategies were tried, and no match was found |
| **ESCALATED** | Part of the video could not be transcribed, so the requested content cannot safely be ruled out |

This prevents incomplete transcript coverage from being incorrectly reported as "not found".

### Reconsider only when the evidence changes

Terminal requests store the `transcript_version` they were judged against.

If a request previously returned `UNFOUND` or `ESCALATED`:

- the same transcript version means the request is skipped
- a changed transcript version means new evidence exists and the request is reconsidered

This is what allows a previously failed chunk to be recovered on a later run without re-running completed requests unnecessarily.

### Coarse retrieval, precise clipping

Search is performed over overlapping transcript windows so content spanning a chunk boundary is still discoverable.

Once a window is matched, Whisper's utterance timestamps are used to narrow it to the sentence-level region that actually answers the request.

If precise narrowing cannot be performed safely, the system falls back to the larger matched window rather than failing the whole request.

### Never repeat expensive work

Several independent reuse checks prevent unnecessary work:

| Check | Avoids |
|---|---|
| completed chunk in `segments` | another transcription call |
| fulfilled request | re-running the strategy ladder |
| unchanged `transcript_version` | reconsidering the same evidence |
| embedding cached by text hash | another embedding API call |
| valid clip already exists for the same time span | another ffmpeg encode |

Clips are identified by their start and end timestamps rather than by the wording of the request. Two differently-worded requests that resolve to the same footage therefore reuse the same clip.

### Partial work never looks complete

Clips are first written to a temporary `*.partial.mp4` file.

Only after the output has been checked for existence, non-zero size and expected duration is it renamed to its final filename.

This means an interrupted ffmpeg process cannot leave behind a half-written file that a later run mistakes for a successful clip.

### Whole-pipeline reruns

The recovery strategy deliberately avoids restarting the entire pipeline.

Each stage resumes independently from persisted state, so recovery is local to the work that actually failed.

Audio chunking is recreated on every run because it is local, deterministic and inexpensive. The expensive work (transcripts, embeddings and clips ) is what is persisted and reused.

The main known limitation is that changing the source video is not currently fingerprinted. If the input video changes while `state.db` is retained, the existing transcript state could become stale; input fingerprinting would be the first correctness improvement I would add.