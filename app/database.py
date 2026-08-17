# app/database.py
"""
All persistent state lives here.

The pipeline's core promise is that completed work is never thrown away, and
that promise is only as good as this file. Two things changed in M0:

1. The database path is now anchored to the project (see config.py) instead
   of the current working directory.
2. start_time / end_time are REAL (floats), because real chunk boundaries are
   not whole numbers and the last chunk is not a whole chunk.
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import DB_PATH


def textHash(text: str):
    """Short stable fingerprint of a piece of text."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def utcNow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    """
    Hand out a connection that always cleans up after itself.

    Commit if the block succeeded, roll back if it raised, close either way.
    Previously every function did open / execute / commit / close by hand,
    which works right up until something raises in the middle - then the
    commit is skipped, the connection is never closed, and you are left
    guessing what actually got written.

    Since the entire project is about surviving failures halfway through,
    "a write either fully happened or fully did not" is a property worth
    buying for six lines of code.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    # Rows can now be read by column name - row["status"] instead of row[3].
    # Positional access breaks silently and wrongly when a column is added;
    # name access breaks loudly, which is the failure mode you want.
    # sqlite3.Row still unpacks like a tuple, so existing code keeps working.
    connection.row_factory = sqlite3.Row

    # Write-ahead logging. If the process is killed mid-write (Ctrl-C, crash,
    # closing the laptop), SQLite can recover the file rather than leaving it
    # corrupt. Cheap insurance for a pipeline designed to be interrupted.
    connection.execute("PRAGMA journal_mode=WAL")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialiseDatabase():
    with connect() as connection:
        # IF NOT EXISTS so re-running never destroys existing state.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS segments (
                chunk_name  TEXT PRIMARY KEY,
                start_time  REAL,
                end_time    REAL,
                status      TEXT NOT NULL,
                transcript  TEXT,
                error       TEXT,
                retry_count INTEGER DEFAULT 0
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS search_windows (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL NOT NULL,
                end_time   REAL NOT NULL,
                transcript TEXT NOT NULL
            )
            """
        )

        # One row per clip request, carrying its whole lifecycle.
        #
        # request_id is derived from the query text (see request_store.py), so
        # a request IS its question. Change the wording and it is a different
        # request with its own state - no comparison logic needed, because
        # there is nothing that can change underneath a key.
        #
        # transcript_version is the interesting column. When a request reaches
        # a terminal state it records the fingerprint of the transcript it was
        # judged against. On a later run we compare: same transcript means
        # asking again would spend money to reach an identical conclusion, so
        # we skip it. A different transcript - because a chunk that previously
        # failed has since succeeded - means the evidence changed and every
        # unfound or escalated request deserves reconsidering.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                request_id         TEXT PRIMARY KEY,
                query              TEXT NOT NULL,
                label              TEXT,
                status             TEXT NOT NULL,
                window_id          INTEGER,
                start_time         REAL,
                end_time           REAL,
                clip_path          TEXT,
                confidence         REAL,
                similarity         REAL,
                attempts           INTEGER NOT NULL DEFAULT 0,
                strategies_tried   TEXT,
                resolution         TEXT,
                transcript_version TEXT,
                updated_at         TEXT
            )
            """
        )

        # Individual spoken utterances, with absolute video timestamps.
        #
        # Whisper returns per-utterance timings, relative to the chunk it was
        # given. Storing them at chunk_start + relative makes them absolute,
        # which is what lets a clip be cut at the sentence that answers the
        # request rather than at the 45-second block that happens to contain
        # it. Without this the finest possible cut is a whole window - about 90
        # seconds - which is a chapter, not a clip.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS utterances (
                chunk_name TEXT NOT NULL,
                idx        INTEGER NOT NULL,
                start_time REAL NOT NULL,
                end_time   REAL NOT NULL,
                text       TEXT NOT NULL,
                PRIMARY KEY (chunk_name, idx)
            )
            """
        )

        # Small key/value store. Currently holds the transcript version the
        # search index was built from, which is what stops stale windows
        # surviving a chunk that failed on one run and succeeded on the next.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        # Embeddings keyed on the HASH OF THE TEXT, not on the window that
        # happens to contain it. Windows get torn down and rebuilt whenever the
        # transcript changes; keying on window id meant re-paying for vectors
        # of text that had not changed at all. Keyed on content, a rebuild is
        # free for every window whose text survived.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT NOT NULL,
                model     TEXT NOT NULL,
                vector    TEXT NOT NULL,
                PRIMARY KEY (text_hash, model)
            )
            """
        )

        # Every choice the agent makes, with the evidence behind it.
        #
        # `signals` holds the numbers the decision was actually made from.
        # A log that records only the verdict is an assertion; one that records
        # the inputs can be audited, and lets you ask "was that call correct?"
        # months later rather than "what did it do?".
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                stage      TEXT NOT NULL,
                request_id TEXT,
                chunk_name TEXT,
                attempt    INTEGER,
                strategy   TEXT,
                signals    TEXT,
                decision   TEXT NOT NULL,
                reasoning  TEXT NOT NULL
            )
            """
        )


# --------------------------------------------------------------------------
# segments
# --------------------------------------------------------------------------

def upsertSegmentBounds(chunk_name: str, start_time: float, end_time: float):
    """
    Record where a chunk sits in the video WITHOUT touching its transcript,
    status or error.

    Boundaries and transcription are separate concerns that are learned at
    different times. We learn exact boundaries when we cut the audio; the
    transcript may already exist from a previous run. Writing boundaries must
    therefore never clobber completed work - hence the narrow UPDATE clause
    below, which only touches the two timing columns.

    This is also what repairs the bad boundaries already in the database:
    on the next run every row gets the real numbers ffmpeg reported, while
    every existing transcript survives untouched.
    """
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO segments (
                chunk_name, start_time, end_time, status, retry_count
            )
            VALUES (?, ?, ?, 'pending', 0)

            ON CONFLICT(chunk_name)
            DO UPDATE SET
                start_time = excluded.start_time,
                end_time   = excluded.end_time
            """,
            (chunk_name, start_time, end_time),
        )


def getSegment(chunk_name: str):
    with connect() as connection:
        return connection.execute(
            """
            SELECT chunk_name, start_time, end_time, status,
                   transcript, error, retry_count
            FROM segments
            WHERE chunk_name = ?
            """,
            (chunk_name,),
        ).fetchone()


def getCompletedSegments():
    with connect() as connection:
        return connection.execute(
            """
            SELECT chunk_name, start_time, end_time, transcript
            FROM segments
            WHERE status = 'completed'
            ORDER BY start_time
            """
        ).fetchall()


def getAllSegments():
    """Every segment regardless of status - used for progress and coverage."""
    with connect() as connection:
        return connection.execute(
            """
            SELECT chunk_name, start_time, end_time, status,
                   transcript, error, retry_count
            FROM segments
            ORDER BY start_time
            """
        ).fetchall()


def saveCompletedSegment(chunk_name: str, start_time: float,
                         end_time: float, transcript: str):
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO segments (
                chunk_name, start_time, end_time,
                status, transcript, error, retry_count
            )
            VALUES (?, ?, ?, 'completed', ?, NULL, 0)

            ON CONFLICT(chunk_name)
            DO UPDATE SET
                status     = 'completed',
                transcript = excluded.transcript,
                error      = NULL
            """,
            (chunk_name, start_time, end_time, transcript),
        )


def saveFailedSegment(chunk_name: str, start_time: float,
                      end_time: float, error: str):
    """
    Mark a chunk failed and increment its attempt counter.

    retry_count is deliberately incremented rather than set, because the agent
    needs to know HOW MANY times something has been tried, not just that it is
    currently broken. Nothing reads it yet - that is M5, where it becomes the
    input to "retry, change parameters, or give up".
    """
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO segments (
                chunk_name, start_time, end_time,
                status, transcript, error, retry_count
            )
            VALUES (?, ?, ?, 'failed', NULL, ?, 1)

            ON CONFLICT(chunk_name)
            DO UPDATE SET
                status      = 'failed',
                error       = excluded.error,
                retry_count = segments.retry_count + 1
            """,
            (chunk_name, start_time, end_time, error),
        )


# --------------------------------------------------------------------------
# search windows
# --------------------------------------------------------------------------

def clearSearchWindows():
    with connect() as connection:
        connection.execute("DELETE FROM search_windows")


def saveSearchWindow(start_time: float, end_time: float, transcript: str):
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO search_windows (start_time, end_time, transcript)
            VALUES (?, ?, ?)
            """,
            (start_time, end_time, transcript),
        )


def getSearchWindows():
    with connect() as connection:
        return connection.execute(
            """
            SELECT id, start_time, end_time, transcript
            FROM search_windows
            ORDER BY start_time
            """
        ).fetchall()


# --------------------------------------------------------------------------
# transcript state - what the agent is allowed to reason from
# --------------------------------------------------------------------------

def transcriptVersion():
    """
    A fingerprint of the current usable transcript.

    Changes when a chunk newly succeeds, when a transcript's text changes, or
    when a chunk's status changes. Does NOT change when unrelated things do -
    so comparing it answers exactly one question: "is there new evidence since
    the last time I decided this?"
    """
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT chunk_name, status, transcript
            FROM segments
            ORDER BY chunk_name
            """
        ).fetchall()

    digest = hashlib.sha256()

    for row in rows:
        digest.update(row["chunk_name"].encode("utf-8"))
        digest.update(row["status"].encode("utf-8"))
        digest.update(textHash(row["transcript"]).encode("utf-8"))

    return digest.hexdigest()[:16]


def getCoverage():
    """
    How much of the video is backed by a usable transcript.

    Returns:
        {"total": float, "covered": float, "ratio": float,
         "gaps": [{"start": float, "end": float, "chunk_name": str}]}

    The gaps list is the important part. "78% covered" tells the agent how much
    it is missing; the gap boundaries tell it WHERE, which is what makes it
    possible to ask "is the thing I failed to find inside a hole?" rather than
    just giving up.
    """
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT chunk_name, start_time, end_time, status
            FROM segments
            ORDER BY start_time
            """
        ).fetchall()

    total = 0.0
    covered = 0.0
    gaps = []

    for row in rows:
        start = row["start_time"] or 0.0
        end = row["end_time"] or 0.0
        span = max(0.0, end - start)
        total += span

        if row["status"] == "completed":
            covered += span
        else:
            gaps.append(
                {
                    "start": start,
                    "end": end,
                    "chunk_name": row["chunk_name"],
                }
            )

    ratio = (covered / total) if total > 0 else 0.0

    return {"total": total, "covered": covered, "ratio": ratio, "gaps": gaps}


# --------------------------------------------------------------------------
# requests
# --------------------------------------------------------------------------

def upsertRequest(request_id: str, query: str, label: str = None):
    """
    Register a request. Existing rows are left alone.

    Because request_id is derived from the query text, "the query changed"
    cannot happen - a changed query is simply a different request_id, and the
    old row stays behind as history. That deletes an entire category of stale
    state without any logic to manage it.
    """
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO requests (
                request_id, query, label, status, attempts, updated_at
            )
            VALUES (?, ?, ?, 'pending', 0, ?)
            ON CONFLICT(request_id) DO NOTHING
            """,
            (request_id, query, label, utcNow()),
        )


def getRequest(request_id: str):
    with connect() as connection:
        return connection.execute(
            "SELECT * FROM requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()


def getAllRequests(request_ids: list[str] = None):
    """
    Every request, or only the ones currently in the requests file.

    Filtering matters for the report: rewording a query creates a new request
    and leaves the old one behind as history. History is worth keeping, but a
    summary of "this run" should show what was actually asked for this run.
    """
    with connect() as connection:
        if request_ids is None:
            return connection.execute(
                "SELECT * FROM requests ORDER BY rowid"
            ).fetchall()

        if not request_ids:
            return []

        placeholders = ",".join("?" for _ in request_ids)
        return connection.execute(
            f"SELECT * FROM requests WHERE request_id IN ({placeholders}) "
            f"ORDER BY rowid",
            request_ids,
        ).fetchall()


def saveUtterances(chunk_name: str, chunk_start: float, segments):
    """
    Store one chunk's utterances with absolute video timestamps.

    `segments` is what Whisper returns under verbose_json: objects with
    .start / .end (relative to the chunk) and .text. Adding chunk_start makes
    them absolute - which is the whole reason the chunker measures real
    boundaries instead of assuming index * 45.

    Replaces any existing rows for the chunk, so a re-transcription overwrites
    rather than accumulating two versions of the same audio.
    """
    with connect() as connection:
        connection.execute(
            "DELETE FROM utterances WHERE chunk_name = ?", (chunk_name,)
        )

        for index, segment in enumerate(segments or []):
            start = float(getattr(segment, "start", 0.0))
            end = float(getattr(segment, "end", 0.0))
            text = (getattr(segment, "text", "") or "").strip()

            if not text:
                continue

            connection.execute(
                """
                INSERT INTO utterances (chunk_name, idx, start_time, end_time, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_name, index, chunk_start + start, chunk_start + end, text),
            )


def getUtterancesInRange(start_time: float, end_time: float):
    """Every utterance overlapping a span, in order."""
    with connect() as connection:
        return connection.execute(
            """
            SELECT chunk_name, idx, start_time, end_time, text
            FROM utterances
            WHERE end_time > ? AND start_time < ?
            ORDER BY start_time
            """,
            (start_time, end_time),
        ).fetchall()


def getFailedSegments():
    """Chunks that have no usable transcript - the holes in coverage."""
    with connect() as connection:
        return connection.execute(
            """
            SELECT chunk_name, start_time, end_time, status, error, retry_count
            FROM segments
            WHERE status != 'completed'
            ORDER BY start_time
            """
        ).fetchall()


# --------------------------------------------------------------------------
# meta + embedding cache
# --------------------------------------------------------------------------

def getMeta(key: str, default=None):
    with connect() as connection:
        row = connection.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()

    return row["value"] if row else default


def setMeta(key: str, value: str):
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, str(value)),
        )


def getCachedEmbedding(text: str, model: str):
    with connect() as connection:
        row = connection.execute(
            "SELECT vector FROM embeddings WHERE text_hash = ? AND model = ?",
            (textHash(text), model),
        ).fetchone()

    return json.loads(row["vector"]) if row else None


def saveCachedEmbedding(text: str, model: str, vector: list[float]):
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO embeddings (text_hash, model, vector) VALUES (?, ?, ?)
            ON CONFLICT(text_hash, model) DO NOTHING
            """,
            (textHash(text), model, json.dumps(vector)),
        )


def saveRequestOutcome(request_id: str, status: str, resolution: str,
                       window_id=None, start_time=None, end_time=None,
                       clip_path=None, confidence=None, similarity=None,
                       strategies_tried=None, attempts=None,
                       transcript_version=None):
    """Write a request's terminal (or in-progress) outcome."""
    with connect() as connection:
        connection.execute(
            """
            UPDATE requests
            SET status = ?, resolution = ?, window_id = ?, start_time = ?,
                end_time = ?, clip_path = ?, confidence = ?, similarity = ?,
                strategies_tried = ?, transcript_version = ?,
                attempts = COALESCE(?, attempts), updated_at = ?
            WHERE request_id = ?
            """,
            (
                status,
                resolution,
                window_id,
                start_time,
                end_time,
                str(clip_path) if clip_path else None,
                confidence,
                similarity,
                json.dumps(strategies_tried) if strategies_tried else None,
                transcript_version,
                attempts,
                utcNow(),
                request_id,
            ),
        )


# --------------------------------------------------------------------------
# decision log
# --------------------------------------------------------------------------

def logDecision(stage: str, decision: str, reasoning: str,
                request_id: str = None, chunk_name: str = None,
                attempt: int = None, strategy: str = None,
                signals: dict = None):
    """
    Record one choice, and the evidence it was made from.

    This is the audit trail the brief asks for. Note that `signals` is stored
    as JSON rather than being flattened into the reasoning string: the prose is
    for a human reading the report, the JSON is for anyone who wants to check
    whether the threshold that produced this decision was the right one.
    """
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO decision_log (
                created_at, stage, request_id, chunk_name,
                attempt, strategy, signals, decision, reasoning
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utcNow(),
                stage,
                request_id,
                chunk_name,
                attempt,
                strategy,
                json.dumps(signals) if signals else None,
                decision,
                reasoning,
            ),
        )


def getDecisionLog(request_id: str = None):
    with connect() as connection:
        if request_id is None:
            return connection.execute(
                "SELECT * FROM decision_log ORDER BY id"
            ).fetchall()

        return connection.execute(
            "SELECT * FROM decision_log WHERE request_id = ? ORDER BY id",
            (request_id,),
        ).fetchall()
