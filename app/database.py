# app/database.py

import sqlite3
from pathlib import Path
import json

DB_PATH = Path("state.db")


def getConnection():
    return sqlite3.connect(DB_PATH)


def initialiseDatabase():
    connection = getConnection()
    #connection is the connection to database(file in this case), cursor is the command we send to do stuff( sql)
    cursor = connection.cursor()
    #if not exists so we dont destroy it on later runs
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS segments (
            chunk_name TEXT PRIMARY KEY,
            start_time INTEGER,
            end_time INTEGER,
            status TEXT NOT NULL,
            transcript TEXT,
            error TEXT,
            retry_count INTEGER DEFAULT 0
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS search_windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time INTEGER NOT NULL,
            end_time INTEGER NOT NULL,
            transcript TEXT NOT NULL,
            embedding TEXT
        )
        """
    )
    #the embedding is a vector ( array of nums) sqlite doesnt have that list so we'll turn into string and store
    #save perma to database file
    connection.commit()
    #close connection
    connection.close()

def getSegment(chunk_name: str):
    #look up a chunk
    connection = getConnection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT chunk_name,
               start_time,
               end_time,
               status,
               transcript,
               error,
               retry_count
        FROM segments
        WHERE chunk_name = ?
        """,
        (chunk_name,),
    )

    row = cursor.fetchone()

    connection.close()

    return row

def getCompletedSegments():
    connection = getConnection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT chunk_name,
               start_time,
               end_time,
               transcript
        FROM segments
        WHERE status = 'completed'
        ORDER BY start_time
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows

def saveCompletedSegment(chunk_name: str,start_time: int,end_time: int,transcript: str):
    connection = getConnection()

    cursor = connection.cursor()
    #if it isnt there then insert if it is make its status completed
    cursor.execute(
        """
        INSERT INTO segments (
            chunk_name,
            start_time,
            end_time,
            status,
            transcript,
            error,
            retry_count
        )
        VALUES (?, ?, ?, 'completed', ?, NULL, 0)

        ON CONFLICT(chunk_name)
        DO UPDATE SET
            status = 'completed',
            transcript = excluded.transcript,
            error = NULL
        """,
        (
            chunk_name,
            start_time,
            end_time,
            transcript,
        ),
    )

    connection.commit()
    connection.close()

def saveFailedSegment(chunk_name: str,start_time: int,end_time: int,error: str):
    connection = getConnection()

    cursor = connection.cursor()
    #retry count is useful for agent decision and handling
    #agent needs to know what failed why how many times we tried
    #if not there add if there update status
    cursor.execute(
        """
        INSERT INTO segments (
            chunk_name,
            start_time,
            end_time,
            status,
            transcript,
            error,
            retry_count
        )
        VALUES (?, ?, ?, 'failed', NULL, ?, 1)

        ON CONFLICT(chunk_name)
        DO UPDATE SET
            status = 'failed',
            error = excluded.error,
            retry_count = segments.retry_count + 1
        """,
        (
            chunk_name,
            start_time,
            end_time,
            error,
        ),
    )

    connection.commit()
    connection.close()

def clearSearchWindows():
    connection = getConnection()
    cursor = connection.cursor()

    cursor.execute("DELETE FROM search_windows")

    connection.commit()
    connection.close()

def saveSearchWindow(
    start_time: int,
    end_time: int,
    transcript: str,
):
    connection = getConnection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO search_windows (
            start_time,
            end_time,
            transcript
        )
        VALUES (?, ?, ?)
        """,
        (
            start_time,
            end_time,
            transcript,
        ),
    )

    connection.commit()
    connection.close()

def getSearchWindows():
    connection = getConnection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id,
               start_time,
               end_time,
               transcript,
               embedding
        FROM search_windows
        ORDER BY start_time
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows

#saving embeddings - cant store list in sqlite so we store it as text
def saveWindowEmbedding(
    window_id: int,
    embedding: list[float],
):
    connection = getConnection()
    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE search_windows
        SET embedding = ?
        WHERE id = ?
        """,
        (
            json.dumps(embedding),
            window_id,
        ),
    )

    connection.commit()
    connection.close()