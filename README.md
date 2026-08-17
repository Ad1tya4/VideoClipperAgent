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

### What's stored

| Table | Holds | Prevents |
|---|---|---|
| `segments` | One row per chunk: real boundaries, status, transcript, error, retry_count | Re-transcribing completed audio; losing progress on a crash |
| `utterances` | Every spoken sentence with absolute video timestamps | 90-second clips; cuts landing mid-word |
| `search_windows` | Overlapping pairs of adjacent chunk transcripts | Topics straddling a chunk boundary matching nothing |
| `embeddings` | Vectors keyed by hash of their text | Re-paying when the index is rebuilt |
| `requests` | Per-request lifecycle: status, clip, confidence, strategies, transcript_version | Re-running a decided request; losing a half-finished batch |
| `decision_log` | Every choice, with its signals as JSON | Unauditable behaviour |
| `meta` | Which transcript version the index was built from | A recovered chunk staying invisible to search |


### Reading the decision log

Prose and signals are stored separately on purpose. The sentence is for a human reading the report; the JSON is for anyone who wants to check whether the threshold that produced the decision was the right one:

```
decision  : escalate
reasoning : "Tried 4 strategies (baseline, expand, widen, retranscribe) with no
             match... Transcript covers only 84.5% of the video — unreadable:
             90-135s (chunk_002.mp3). The content cannot be ruled out."
signals   : {"best_similarity": 0.19, "llm_confidence": 0.05,
             "coverage_ratio": 0.845, "gaps": ["90-135s (chunk_002.mp3)"],
             "candidates_seen": 5}
```

Query it directly:

```bash
cd app
python -c "import database; [print(dict(r)) for r in database.getDecisionLog()]"
```

The log is **append-only across runs**, so a request that was escalated on one run and resolved on the next shows both, with the evidence for each.

### A finding from real data

On a real run the correct match scored a cosine similarity of **0.271**  and the LLM verifier confirmed it at **0.99** confidence, correctly. Any naive threshold on similarity would have thrown away a right answer.

Embedding similarity is a good *ranking* signal and a terrible *absolute* one. That's why the LLM gets a vote, and why `route()` never decides anything on similarity alone.

---

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
</p>---