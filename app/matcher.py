import json
import math
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from database import (
    clearSearchWindows,
    getCompletedSegments,
    getSearchWindows,
    saveSearchWindow,
    saveWindowEmbedding,
)


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
VERIFICATION_MODEL = "gpt-5.6-terra"

TOP_K = 5

#Building windows
'''
window 0 = chunks 0 + 1
window 1 = chunks 1 + 2
window 2 = chunks 2 + 3
window 3 = chunks 3 + 4
'''

def buildSearchWindows():
    segments = getCompletedSegments()

    if not segments:
        return

    existing_windows = getSearchWindows()
    #we dont regenerate everytime
    if existing_windows:
        return

    if len(segments) == 1:
        _, start_time, end_time, transcript = segments[0]

        saveSearchWindow(
            start_time=start_time,
            end_time=end_time,
            transcript=transcript,
        )

        return

    for index in range(len(segments) - 1):
        current = segments[index]
        next_segment = segments[index + 1]

        current_name, current_start, current_end, current_text = current
        next_name, next_start, next_end, next_text = next_segment

        combined_transcript = (
            f"{current_text}\n"
            f"{next_text}"
        )

        saveSearchWindow(
            start_time=current_start,
            end_time=next_end,
            transcript=combined_transcript,
        )

#An embedding is a learned dense vector of numbers that represents the meaning of the text.
#each position is not a word like one hot or bag-of-words the model has learned many abstract semantic features
#then we do cosine similarity of the embeddings to see which windows are important instead of sending whole transcript to llm which is long and costly
def createEmbedding(text: str):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding
#we create embeddings and store it and check so if its already computed we dont do it again (necessary calc and pay)
def embedSearchWindows():
    windows = getSearchWindows()

    for window in windows:
        window_id, start_time, end_time, transcript, embedding_json = window

        if embedding_json is not None:
            continue

        embedding = createEmbedding(transcript)

        saveWindowEmbedding(
            window_id=window_id,
            embedding=embedding,
        )

        print(
            f"[EMBEDDED] window {window_id} "
            f"({start_time}-{end_time}s)"
        )
#to see if embeddings are in same dir/ if they simmilar
#1.0 will be very simmilar direction, 0 unrelated
def cosineSimilarity(
    vector_a: list[float],
    vector_b: list[float],
):
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)

#now we retrieve top k candidates
def retrieveCandidates(
    query: str,
    top_k: int = TOP_K,
):
    query_embedding = createEmbedding(query)

    windows = getSearchWindows()

    candidates = []

    for window in windows:
        (
            window_id,
            start_time,
            end_time,
            transcript,
            embedding_json,
        ) = window

        if embedding_json is None:
            continue # skip it

        transcript_embedding = json.loads(
            embedding_json
        )

        similarity = cosineSimilarity(
            query_embedding,
            transcript_embedding,
        )

        candidates.append(
            {
                "window_id": window_id,
                "start_time": start_time,
                "end_time": end_time,
                "transcript": transcript,
                "similarity": similarity,
            }
        )

    candidates.sort(
        key=lambda candidate: candidate["similarity"],
        reverse=True,
    )
    #llm will see these - dont wanna send it whole vid. 5 min vid now but this scales nicely
    return candidates[:top_k]

#we want structured model output not just oh i think this is good
#this structure is pydantic we defien the schema - Pydantic is a Python library for defining and validating structured data
#we want answers in this format
class MatchVerification(BaseModel):
    matched: bool
    window_id: int | None
    confidence: float
    reason: str
#format the top k windows
def formatCandidates(candidates):
    sections = []

    for candidate in candidates:
        section = f"""
    WINDOW ID: {candidate["window_id"]}
    START TIME: {candidate["start_time"]} seconds
    END TIME: {candidate["end_time"]} seconds
    EMBEDDING SIMILARITY: {candidate["similarity"]:.3f}

    TRANSCRIPT:
    {candidate["transcript"]}
    """

        sections.append(section)

    return "\n---\n".join(sections)

#llm verification of cosine similarity to be sure to get another opinion
def verify_candidates_with_llm(
    query: str,#what the user asks
    candidates: list[dict],#topk candidates
):
    if not candidates:
        return MatchVerification(
            matched=False,
            window_id=None,
            confidence=0.0,
            reason="No transcript candidates were available.",
        )

    candidate_text = formatCandidates(candidates)

    response = client.responses.parse(
        model=VERIFICATION_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a semantic verifier for a video clipping system. "
                    "The user provides a natural-language request describing "
                    "content they want to find in a video. "
                    "You are given candidate transcript windows retrieved using "
                    "embedding similarity. "
                    "\n\n"
                    "Determine whether any candidate genuinely contains the "
                    "content requested by the user. Match meaning, not exact "
                    "wording. Synonyms, paraphrases and equivalent business "
                    "language should count as matches. "
                    "\n\n"
                    "Do not select a candidate merely because it shares a few "
                    "words with the request. The underlying subject and meaning "
                    "must correspond. "
                    "\n\n"
                    "If none of the candidates meaningfully matches the request, "
                    "return matched=false and window_id=null. "
                    "\n\n"
                    "Confidence must be between 0 and 1."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CLIP REQUEST:\n{query}\n\n"
                    f"CANDIDATES:\n{candidate_text}"
                ),
            },
        ],
        text_format=MatchVerification,
    )

    return response.output_parsed

def findMatch(query: str):
    buildSearchWindows()
    embedSearchWindows()

    candidates = retrieveCandidates(query)

    print("\nTOP RETRIEVAL CANDIDATES")

    for candidate in candidates:
        print(
            f"Window {candidate['window_id']} | "
            f"{candidate['start_time']}-{candidate['end_time']}s | "
            f"similarity={candidate['similarity']:.3f}"
        )
    #wchemea makes it return best one
    verification = verify_candidates_with_llm(
        query=query,
        candidates=candidates,
    )

    return verification, candidates