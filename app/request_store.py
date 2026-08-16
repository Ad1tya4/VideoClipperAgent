# app/request_store.py
"""
Load clip requests from disk and register them in the database.

The brief asks for "a set of clip requests", not one interactive question, and
the difference matters: a batch can be half-finished. Putting requests in the
database means a run that dies after fulfilling three of five resumes at
request four rather than starting over.

File format (requests.json at the project root):

    [
      {"query": "the part where they discuss budget constraints"},
      {"query": "the section about Q3 results", "label": "q3"}
    ]

A request's identity IS its query text - request_id is a hash of it. `label` is
cosmetic, used to make the report readable, and has no effect on state.

Keying on the text means rewording a query produces a different request rather
than silently overwriting the old one's result, and it means there is no
"has the query changed under this id?" check to write, get wrong, or forget.
Same rule the clip filenames follow: identify a thing by what it is.
"""

import json

from config import REQUESTS_PATH
from database import textHash, upsertRequest

EXAMPLE_REQUESTS = [
    {"label": "youtube", "query": "the part where he talks about YouTube"},
    {"label": "absent", "query": "a detailed discussion of medieval Icelandic property law"},
]


def requestId(query: str):
    return f"q-{textHash(query)[:10]}"


def writeExampleRequests(path=REQUESTS_PATH):
    path.write_text(json.dumps(EXAMPLE_REQUESTS, indent=2), encoding="utf-8")
    return path


def loadRequestsFile(path=REQUESTS_PATH):
    """
    Read and validate the requests file.

    Validation fails loudly and early. A malformed requests file caught at load
    time is a typo; the same file caught halfway through a run, after money has
    been spent transcribing, is a wasted run.
    """
    if not path.exists():
        raise FileNotFoundError(f"No requests file at {path}.")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error

    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list of requests.")

    requests = []
    seen = set()

    for index, entry in enumerate(raw):
        if isinstance(entry, str):
            entry = {"query": entry}

        if not isinstance(entry, dict):
            raise ValueError(f"Request {index} in {path} is not an object or string.")

        query = (entry.get("query") or "").strip()

        if not query:
            raise ValueError(f"Request {index} in {path} has no 'query'.")

        request_id = requestId(query)

        # The same question asked twice is one request, not two.
        if request_id in seen:
            continue

        seen.add(request_id)
        requests.append(
            {
                "request_id": request_id,
                "query": query,
                "label": (entry.get("label") or "").strip() or None,
            }
        )

    if not requests:
        raise ValueError(f"{path} contains no requests.")

    return requests


def syncRequests(path=REQUESTS_PATH):
    """Register every request from the file. Returns the active request ids."""
    requests = loadRequestsFile(path)

    for request in requests:
        upsertRequest(
            request_id=request["request_id"],
            query=request["query"],
            label=request["label"],
        )

    return [request["request_id"] for request in requests]
