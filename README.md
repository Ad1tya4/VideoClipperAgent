# Video Clipper Agent

Cuts clips out of a long recording from natural-language requests — and, more to the point, **decides what to do when it can't find something.**

Transcription fails partway through. A request doesn't match. The interesting question isn't how to search a transcript, it's what an agent should do when the search comes back empty: retry with different parameters, give up and say so, or admit it doesn't know. This project's answer to that question is the thing it's actually about.

---

## What it does

- Splits a video's audio into chunks and transcribes them, **resuming from wherever it stopped** if a run is interrupted or a chunk fails
- Searches the transcript for each clip request using embeddings, verified by an LLM
- When a request doesn't match, **escalates through four strategies** before giving a verdict
- Distinguishes **"it isn't there"** from **"I couldn't read part of the video, so I don't know"** — these are different claims and are reported differently
- Cuts frame-accurate clips at sentence boundaries, not chunk boundaries
- **Never repeats expensive work**: transcripts, embeddings and finished clips all survive across runs
- Logs every decision with the numbers behind it

---

## Quick start

### Prerequisites

**Python 3.10+** and **ffmpeg** (which includes `ffprobe`). ffmpeg is not a pip package:

| Platform | Install |
|---|---|
| Windows | `winget install ffmpeg` — or [download](https://ffmpeg.org/download.html) and add to PATH |
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

Drop any video into `data/`. **That's it** — the agent finds it automatically, whatever it's called:

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

`label` is cosmetic — it just makes the report readable. A request's real identity is a hash of its `query` text, so rewording a question makes it a new request rather than silently overwriting the old one's result.

### Run it

```bash
cd app
python main.py
```

Clips land in `app/clips/`. All commands run from inside `app/`.

---

## Seeing the failure recovery

The whole point of the project is what happens when transcription fails, so failure is injectable with a flag rather than hidden behind a commented-out line. Two runs:

```bash
cd app

python main.py --fault chunk_002.mp3     # run 1: that chunk fails
python main.py                           # run 2: recovers
```

### Run 1 — a chunk fails

```
Transcribing...
[SUCCESS] chunk_000.mp3 (6 utterances)
[SUCCESS] chunk_001.mp3 (12 utterances)
[FAILED]  chunk_002.mp3 - Simulated transcription timeout (--fault)
[SUCCESS] chunk_003.mp3 (10 utterances)
...

Transcript coverage: 245.6s of 290.6s (84.5%)
  GAP  01:30-02:15  (chunk_002.mp3)

Requests: 1 escalated, 1 fulfilled
```

The YouTube request matched on the first strategy and produced a clip. The other one escalated:

```
[ESCALATED]
  asked:   "a detailed discussion of medieval Icelandic property law"
  why:     Tried 4 strategies (baseline, expand, widen, retranscribe) with no
           match, and re-transcription could not repair every gap. Transcript
           covers only 84.5% of the video - unreadable: 90-135s
           (chunk_002.mp3). The content cannot be ruled out, because part of
           the recording was never searched. Needs a human.
```

It tried re-transcribing the failed chunk (which failed again — the injected fault applies to retries too), then refused to claim the content was absent, because it couldn't see 45 seconds of the video.

### Run 2 — recovery

```
Transcribing...
[SKIP]    chunk_000.mp3 already transcribed
[SKIP]    chunk_001.mp3 already transcribed
[WORK]    chunk_002.mp3 [01:30 - 02:15]
[SUCCESS] chunk_002.mp3 (17 utterances)
[SKIP]    chunk_003.mp3 already transcribed
[SKIP]    chunk_004.mp3 already transcribed
[SKIP]    chunk_005.mp3 already transcribed
[SKIP]    chunk_006.mp3 already transcribed

Transcript coverage: 290.6s of 290.6s (100.0%)
  No gaps - the whole recording is searchable.

[SKIP]    q-25a8131b93: already fulfilled - the clip exists
[WORK]    q-cdf5770259: previously escalated, but the transcript has changed
                        since (a chunk that was unreadable is now
                        transcribed) - reconsidering
[UNFOUND] q-cdf5770259 (strategies: baseline, expand, widen)
```

Six chunks skipped, one transcribed. The fulfilled request cost nothing. The escalated request was reconsidered **because its evidence changed** — and this time reached a different, stronger verdict:

```
[UNFOUND]
  why:     Tried 3 strategies (baseline, expand, widen) with no match.
           Transcript covers 100.0% of the video with no gaps, so the whole
           recording was searched: this content is not in the video.
```

Three strategies, not four — because with complete coverage the agent **declined** to re-transcribe:

```
retranscribe -> decline
  "Declined to re-transcribe: the transcript already covers 100% of the video,
   so re-reading audio cannot surface content that was missed. Spending money
   here could not change the outcome."
```

None of that is special-cased. No code path knows what "run 2" is. It falls out of one comparison — *does the transcript I'm looking at differ from the one I decided against?* — and one guard: *can this action possibly help?*

---

## The approach, and why

### 1. Two kinds of "no"

The decision I'd defend first. When a request can't be matched, there are two very different things that might be true:

| | Claim | Trigger | What you do with it |
|---|---|---|---|
| **UNFOUND** | "It isn't in the video." | Coverage complete, all strategies tried | Trust it, move on |
| **ESCALATED** | "I don't know — I couldn't read part of it." | Coverage incomplete after re-transcription failed | Check the named gap yourself |

Collapsing these into one "not found" reports incomplete knowledge as a conclusion. The agent tracks *which seconds of video it was able to read*, so it can tell the difference — and it names the exact gap when it can't.

### 2. The escalation ladder

A request that doesn't match isn't abandoned after one try. It escalates through strategies that each attack a **different reason for failing** — which is why they're ordered this way rather than being four attempts at the same thing:

| Strategy | The failure it addresses |
|---|---|
| **baseline** | None — the normal path. Top-5 windows by embedding similarity, verified by an LLM. |
| **expand** | **Vocabulary mismatch.** The user asks for "budget constraints"; the speaker said "we can't afford it". The model rewrites the request into 4 other phrasings and each gets its own shot. The most common real-world miss. |
| **widen** | **Semantic blur.** Embeddings are weak on names, acronyms and numbers — "Q3" and "Q4" are nearly identical vectors and entirely different facts. Top-15 plus a literal word-overlap pass catches what meaning-based search rounds off. |
| **retranscribe** | **Missing evidence.** The content may be in audio that was never successfully read. Re-reads *only* the failed chunks, with the clip request as a vocabulary hint and a different temperature. |

Then the terminal decision above.

**The guard I care most about** is on strategy 4. It runs only if the transcript actually has holes. With complete coverage, re-transcribing cannot surface content that was already fully read, so the agent declines and logs why. Knowing which actions are *pointless* is worth more than being willing to try everything.

### 3. The LLM judges meaning; plain Python decides what happens next

```
The LLM answers   "does this transcript window mean what was asked for?"
                  — a semantic judgement, which is what it's good at.

route() answers   "given that answer, what should happen next?"
                  — a control-flow decision.
```

Asking a model to decide *retry / skip / escalate* would produce a choice I couldn't reproduce, couldn't unit test, and could only explain by quoting the model's own account of itself. `route()` in `agent.py` is a pure function of four plain values — no database, no video, no API key needed to call it:

| Match? | Confidence | Strategies left | Decision |
|---|---|---|---|
| yes | ≥ 0.55 | — | **accept** |
| yes | < 0.55 | yes | **hold** — a weak match is worth *beating*, not discarding. Kept as a fallback while a stronger strategy runs. |
| yes | < 0.55 | no | **accept, flagged** — better to return something marked *review me* than nothing |
| no | — | yes | **next_strategy** |
| no | — | no | **exhausted** → coverage decides which kind of "no" |

Same inputs, same decision, same sentence of reasoning. Every time.

### 4. Retrieval is coarse; selection is fine

Search operates on overlapping ~90-second windows, because when deciding *where to look*, recall matters more than precision — a topic straddling a chunk boundary must not fall between two windows and match neither.

But a 90-second file is a chapter, not a clip. So once a window is chosen, the model gets a second, much cheaper question: *of these numbered sentences, which ones answer the request?* Whisper already computes per-utterance timestamps — the plain-text response just throws them away — so this costs nothing extra to enable.

A 90-second window becomes a 35-second clip cut at sentence boundaries.

Narrowing **degrades rather than breaks**: if utterance timings are missing, or the model returns an out-of-range index, it falls back to the full window with a logged reason. A precision feature that takes down the pipeline when it can't run is worse than no precision feature.

### 5. Never repeating work — five independent gates

| # | Gate | Keyed on | Saves |
|---|---|---|---|
| 1 | Chunk already transcribed | `segments.status` | a Whisper call |
| 2 | Request already fulfilled | `requests.status` | the entire ladder |
| 3 | Terminal request, evidence unchanged | `requests.transcript_version` | the entire ladder |
| 4 | Text already embedded | `sha256(text)` | an embedding call |
| 5 | Span already cut and valid | clip filename | an ffmpeg re-encode |

Gates 4 and 5 are **content-addressed** — keyed on what the thing *is*, not on the row or the request that produced it.

That distinction fixed a real bug. Embeddings were originally stored on the window row, and the search index was only built once ("if windows exist, do nothing"). So a chunk that failed on one run and succeeded on the next never entered the index — **the recovered content stayed permanently unfindable.** The fix wasn't smarter invalidation. It was making the rebuild cheap enough to do unconditionally: windows are string concatenation, and embeddings keyed by text hash survive the rebuild, so re-indexing costs zero API calls for every window whose text didn't change.

> When a cache causes correctness problems, make the recompute cheap rather than making the invalidation clever. Invalidation logic is where stale-data bugs live.

The same principle names the clips. Two differently-worded requests that resolve to the same footage produce the same filename, so the second is recognised as already done rather than writing a duplicate file.

### 6. When does the whole pipeline re-run?

**Never.** Every stage resumes independently. A full re-run is a design failure, not a recovery strategy — the only thing that should invalidate everything is the *input* changing, and that's a known gap (see limitations).

The one thing deliberately redone every run is **audio chunking**: local, deterministic, free, no network, no billing. Re-deriving a free artefact isn't what "re-processing" means. Caching it would buy two seconds in exchange for an invalidation problem.

### 7. Partial work must never look complete

Clips are written to `*.partial.mp4` and renamed only after verification passes, so a file at the final path is always a finished file. An interrupted run can't leave a half-written clip that the next run mistakes for completed work.

And verification is real: after ffmpeg exits 0, the output is probed for existence, non-zero size, and correct duration. *"The command didn't error"* and *"the clip is correct"* are different claims.

### 8. Structured output guarantees shape, not truth

Pydantic guarantees the model returns *an integer*. It cannot guarantee the integer refers to something real. A model can return a well-formed `window_id` it was never shown, and cutting on it would produce a clip from an arbitrary point in the video with no error anywhere to explain it. So every identifier that indexes into real data is validated before use — window ids against the candidate list, utterance indices against the list shown, clip files against ffprobe.

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

Boundaries come from ffmpeg's own manifest (`-segment_list`), not from `index × 45`. ffmpeg cuts at the next packet boundary, and it can't make the last chunk 45 seconds long when only 20 seconds remain — the arithmetic version claimed the video was 315 seconds when it was 290.6. Every clip's cut point is `chunk_start + time_within_chunk`, and coverage is computed from these numbers, so an agent reasoning from invented boundaries makes confident wrong decisions.

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

On a real run the correct match scored a cosine similarity of **0.271** — and the LLM verifier confirmed it at **0.99** confidence, correctly. Any naive threshold on similarity would have thrown away a right answer.

Embedding similarity is a good *ranking* signal and a terrible *absolute* one. That's why the LLM gets a vote, and why `route()` never decides anything on similarity alone.

---

## Known limitations

Things I found, understood, and chose not to fix before the deadline.

- **No input fingerprint.** Replace the video in `data/` without deleting `state.db` and the old video's transcripts are reused against the new audio — silently wrong, no error. A hash of `(file size, mtime, chunk length, audio params)` compared against a stored value would catch it in about ten lines. This is the gap I'd close first.
- **Errors aren't classified.** A network timeout and an invalid API key are handled identically. Both are recorded and survived, but a permanent error still burns one retry per chunk instead of stopping immediately. A circuit breaker — *N consecutive failures of the same class, stop and escalate* — is the fix.
- **Thresholds are hand-tuned on one video.** `0.55` confidence, `8s` minimum clip, `0.999` coverage. These are judgement calls, not learned values, and I have no evidence they generalise.
- **The lexical pass is exact-match only.** No fuzzy matching, so a misheard proper noun is still missed. It was meant to use `rapidfuzz`; the standard-library version shipped instead to avoid a dependency.
- **Windows are a fixed two chunks wide.** Content spanning more than ~90 seconds can't be matched as a single unit.
- **One clip per request.** If a topic is discussed in three separate places, only the best match is returned.
- **Transcription is sequential.** Chunks are independent and could run concurrently; on a long video that's most of the wall-clock time.
- **No automated tests.** `route()` is a pure function specifically so it could be tested, and it isn't. That's the most embarrassing item here, and the cheapest to fix.

---

## What I'd do next, with more time

**First, and in this order:**

1. **Input fingerprinting.** The one known correctness bug. Ten lines, and it makes "when do you re-run the whole pipeline?" a complete answer rather than a mostly-complete one.
2. **Tests for `route()`.** The decision function was built as a pure function precisely so it could be tested without a database, a video, or an API key. Every row of that truth table should be a test case, plus the terminal `unfound`/`escalated` split. Maybe 40 lines.
3. **Error classification and a circuit breaker.** Split failures into transient (timeout, 429, 5xx → retry with backoff) and permanent (auth, unsupported format → don't retry at all). Stop the run after N consecutive same-class failures rather than failing forty chunks identically against a dead API key.

**Then, to make it better rather than more correct:**

4. **Parallel transcription.** Chunks are independent. On an hour-long recording this is nearly all the runtime.
5. **Multiple matches per request.** Return every region above threshold instead of only the best one — "find every time they mention the deadline" is a natural request the current design can't answer.
6. **Hierarchical retrieval.** Coarse windows to locate the region, fine windows within it, so long discussions and brief mentions are both matchable. The fixed two-chunk width is a compromise between them.
7. **Cost accounting in the report.** Token and API call counts per request, per strategy. The agent already decides when *not* to spend money; it should be able to show what that saved.
8. **Threshold tuning on labelled data.** Twenty requests against three videos with known answers would turn `0.55` from a guess into a measurement — and would show whether the `hold` behaviour actually helps or just delays.
9. **Speaker diarisation.** "Find where *Sarah* talks about the budget" is a natural request that needs speaker labels the current transcript doesn't carry.

**And the honest one:** I'd re-record the demo against a longer, messier video. Everything here works on a 4:50 recording with clean audio. The failure modes this project is *about* — timeouts, partial transcripts, ambiguous matches — get more interesting at an hour, and I don't have evidence for how it behaves there.

---

## Cost

A 5-minute video costs roughly £0.03 to process from scratch: seven Whisper calls, a handful of embeddings, and one or two LLM verifications per request. Re-runs cost close to nothing, because the five gates above mean the only work done is work that hasn't been done before.
