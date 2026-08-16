# app/matcher.py
"""
Finding a clip request in the transcript.

Two-stage retrieval:

  1. Embeddings rank every transcript window against the request. Cheap, and
     it never has to look at the whole transcript.
  2. An LLM reads only the top few and decides whether any of them genuinely
     means what was asked. Expensive, so it only ever sees a handful.

Worth knowing, because it shapes the whole design: on a real run this project's
correct match scored a cosine similarity of 0.271 and the verifier confirmed it
at 0.99 confidence. Any naive threshold on similarity would have discarded a
correct answer. **Embedding similarity is a good ranking signal and a terrible
absolute one** - which is exactly why the LLM gets a vote, and why the router in
agent.py never decides anything on similarity alone.

This module provides the retrieval strategies. It does not decide which one to
use, or when to stop. That is agent.py's job.
"""

import json
import math
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from config import (
    EMBEDDING_MODEL,
    QUERY_EXPANSIONS,
    TOP_K_BASELINE,
    VERIFICATION_MODEL,
)
from database import (
    clearSearchWindows,
    getCachedEmbedding,
    getCompletedSegments,
    getMeta,
    getSearchWindows,
    saveCachedEmbedding,
    saveSearchWindow,
    setMeta,
    transcriptVersion,
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Two chunks are treated as adjacent if their boundaries meet within this many
# seconds. Boundaries come from ffmpeg as floats, so they will not be equal.
ADJACENCY_TOLERANCE = 0.5


# --------------------------------------------------------------------------
# the search index
# --------------------------------------------------------------------------

def buildSearchWindows():
    """
    Glue adjacent transcribed chunks into overlapping windows.

    Overlapping, because a topic that straddles a chunk boundary would
    otherwise be split across two windows and match neither well.

    NEVER ACROSS A GAP. If chunk 2 failed to transcribe, chunks 1 and 3 are
    adjacent in the table but forty-five seconds apart in the video. Gluing
    them would produce a window claiming to cover 45-180s whose text is
    missing the middle - and the agent would then reason about coverage using
    a window that lies about what it contains. Runs of contiguous chunks are
    windowed independently, so a gap stays visible as a gap.
    """
    segments = getCompletedSegments()

    if not segments:
        return 0

    # Split into runs of chunks that actually touch in the video.
    runs = []
    current = [segments[0]]

    for previous, segment in zip(segments, segments[1:]):
        if abs(segment["start_time"] - previous["end_time"]) <= ADJACENCY_TOLERANCE:
            current.append(segment)
        else:
            runs.append(current)
            current = [segment]

    runs.append(current)

    written = 0

    for run in runs:
        if len(run) == 1:
            saveSearchWindow(run[0]["start_time"], run[0]["end_time"],
                             run[0]["transcript"])
            written += 1
            continue

        for first, second in zip(run, run[1:]):
            saveSearchWindow(
                first["start_time"],
                second["end_time"],
                f"{first['transcript']}\n{second['transcript']}",
            )
            written += 1

    return written


def ensureSearchIndex(force: bool = False):
    """
    Make sure the windows on disk reflect the transcript on disk.

    The original version of this said "if windows exist, do nothing", which
    meant a chunk that failed on one run and succeeded on the next never made
    it into the index - the recovered content stayed permanently unfindable.

    The fix is not smarter invalidation, it is cheap rebuilding: windows are
    string concatenation, and embeddings are cached by text hash, so tearing
    the index down and rebuilding it costs no API calls for any window whose
    text did not change. Rebuild aggressively; cache only what is expensive.
    """
    version = transcriptVersion()
    built_from = getMeta("search_index_version")

    if not force and built_from == version and getSearchWindows():
        return False

    clearSearchWindows()
    buildSearchWindows()
    setMeta("search_index_version", version)
    return True


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------

def createEmbedding(text: str):
    """
    Embed text, reusing a cached vector when we have embedded this exact text
    before. Cached by content hash, so it survives windows being rebuilt.
    """
    cached = getCachedEmbedding(text, EMBEDDING_MODEL)

    if cached is not None:
        return cached

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    vector = response.data[0].embedding
    saveCachedEmbedding(text, EMBEDDING_MODEL, vector)

    return vector


def cosineSimilarity(vector_a, vector_b):
    """1.0 means pointing the same way, 0.0 means unrelated."""
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot / (magnitude_a * magnitude_b)


# --------------------------------------------------------------------------
# retrieval strategies
# --------------------------------------------------------------------------

def scoreWindows(queries: list[str]):
    """
    Score every window against one or more phrasings of the request.

    A window keeps its BEST score across the phrasings, and records which
    phrasing produced it - so the log can say "this was found by the reworded
    query, not the original", which is the evidence that expansion earned its
    cost.
    """
    query_vectors = [(q, createEmbedding(q)) for q in queries]
    candidates = []

    for window in getSearchWindows():
        window_vector = createEmbedding(window["transcript"])

        best_score = 0.0
        best_query = queries[0]

        for query, query_vector in query_vectors:
            score = cosineSimilarity(query_vector, window_vector)
            if score > best_score:
                best_score = score
                best_query = query

        candidates.append(
            {
                "window_id": window["id"],
                "start_time": window["start_time"],
                "end_time": window["end_time"],
                "transcript": window["transcript"],
                "similarity": best_score,
                "matched_query": best_query,
            }
        )

    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates


def retrieveCandidates(query: str, top_k: int = TOP_K_BASELINE):
    return scoreWindows([query])[:top_k]


def retrieveExpanded(queries: list[str], top_k: int):
    return scoreWindows(queries)[:top_k]


class QueryExpansion(BaseModel):
    phrasings: list[str]


def expandQuery(query: str, count: int = QUERY_EXPANSIONS):
    """
    Ask the model for alternative ways the same idea might have been said out
    loud.

    This is the single most common reason a real clip request misses: the user
    asks for "budget constraints" and the speaker said "we can't afford it".
    Embeddings handle paraphrase well but not perfectly, and expanding the
    query attacks the mismatch from the other side - several phrasings, each
    given its own shot at the transcript.
    """
    response = client.responses.parse(
        model=VERIFICATION_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You rewrite requests about video content into alternative "
                    "phrasings, to improve transcript search. Given a request, "
                    f"produce {count} different ways the same idea might "
                    "actually be SPOKEN ALOUD in a recording. Use plain, "
                    "conversational language and likely synonyms. Do not "
                    "repeat the original wording. Return only the phrasings."
                ),
            },
            {"role": "user", "content": query},
        ],
        text_format=QueryExpansion,
    )

    phrasings = [p.strip() for p in response.output_parsed.phrasings if p.strip()]
    return phrasings[:count]


# --------------------------------------------------------------------------
# lexical fallback
# --------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "about", "is", "are", "was", "were", "be", "been", "it", "this",
    "that", "where", "when", "he", "she", "they", "we", "i", "you", "his",
    "her", "their", "part", "section", "bit", "talks", "talk", "discuss",
    "discusses", "discussion", "find", "give", "me", "show", "clip", "video",
    "speaks", "speak", "spoken", "says", "said", "mentions", "mentioned",
}


def contentWords(text: str):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def lexicalCandidates(query: str, top_k: int):
    """
    Score windows by literal word overlap instead of by meaning.

    Embeddings are weak exactly where this is strong: names, product codes,
    acronyms, numbers. "Q3" and "Q4" are nearly identical vectors and entirely
    different facts. A word either appears or it does not, so this catches the
    cases semantic search rounds off.

    Deliberately built on the standard library rather than a fuzzy-matching
    dependency: the score is "what fraction of the request's meaningful words
    appear in this window", which is a sentence anyone can check.
    """
    wanted = contentWords(query)

    if not wanted:
        return []

    candidates = []

    for window in getSearchWindows():
        present = contentWords(window["transcript"])
        overlap = wanted & present

        if not overlap:
            continue

        candidates.append(
            {
                "window_id": window["id"],
                "start_time": window["start_time"],
                "end_time": window["end_time"],
                "transcript": window["transcript"],
                "similarity": len(overlap) / len(wanted),
                "matched_query": query,
                "matched_words": sorted(overlap),
            }
        )

    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates[:top_k]


def mergeCandidates(*groups, limit: int):
    """Combine candidate lists, keeping each window once at its best score."""
    best = {}

    for group in groups:
        for candidate in group:
            existing = best.get(candidate["window_id"])
            if existing is None or candidate["similarity"] > existing["similarity"]:
                best[candidate["window_id"]] = candidate

    merged = sorted(best.values(), key=lambda c: c["similarity"], reverse=True)
    return merged[:limit]


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

class MatchVerification(BaseModel):
    matched: bool
    window_id: int | None
    confidence: float
    reason: str


def formatCandidates(candidates):
    sections = []

    for candidate in candidates:
        sections.append(
            f"WINDOW ID: {candidate['window_id']}\n"
            f"START: {candidate['start_time']:.1f}s   "
            f"END: {candidate['end_time']:.1f}s\n"
            f"RETRIEVAL SCORE: {candidate['similarity']:.3f}\n"
            f"TRANSCRIPT:\n{candidate['transcript']}"
        )

    return "\n---\n".join(sections)


def verifyCandidates(query: str, candidates: list[dict]):
    """
    Ask the model whether any candidate genuinely contains what was requested.

    The prompt pushes hard against false positives, because in this system a
    wrong match is worse than no match: "not found" is an honest answer a user
    can act on, while a confidently wrong clip wastes their time and hides the
    failure.
    """
    if not candidates:
        return MatchVerification(
            matched=False, window_id=None, confidence=0.0,
            reason="No transcript candidates were available to check.",
        )

    response = client.responses.parse(
        model=VERIFICATION_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a semantic verifier for a video clipping system. "
                    "The user describes content they want found in a video. "
                    "You are shown candidate transcript windows retrieved by "
                    "search.\n\n"
                    "Decide whether any candidate genuinely contains the "
                    "requested content. Match meaning, not wording - synonyms, "
                    "paraphrases and equivalent phrasing all count.\n\n"
                    "Do NOT select a candidate merely because it shares a few "
                    "words with the request. The underlying subject must "
                    "correspond. If nothing meaningfully matches, return "
                    "matched=false and window_id=null - a clean 'not found' is "
                    "more useful than a wrong clip.\n\n"
                    "window_id must be one of the ids shown. Confidence is "
                    "between 0 and 1 and should reflect how sure you are that "
                    "this window contains what was asked for."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CLIP REQUEST:\n{query}\n\n"
                    f"CANDIDATES:\n{formatCandidates(candidates)}"
                ),
            },
        ],
        text_format=MatchVerification,
    )

    return response.output_parsed
