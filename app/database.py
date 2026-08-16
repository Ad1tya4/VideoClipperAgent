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

import json
import sqlite3
from contextlib import contextmanager

from config import DB_PATH


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
                transcript TEXT NOT NULL,
                embedding  TEXT
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
            SELECT id, start_time, end_time, transcript, embedding
            FROM search_windows
            ORDER BY start_time
            """
        ).fetchall()


def saveWindowEmbedding(window_id: int, embedding: list[float]):
    """SQLite has no vector type, so the list is stored as a JSON string."""
    with connect() as connection:
        connection.execute(
            "UPDATE search_windows SET embedding = ? WHERE id = ?",
            (json.dumps(embedding), window_id),
        )
