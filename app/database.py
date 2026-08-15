# app/database.py

import sqlite3
from pathlib import Path


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