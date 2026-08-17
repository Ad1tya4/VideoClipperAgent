# app/agent.py
"""
The decision layer - the part that makes this an agent rather than a search box.

THE CENTRAL DESIGN CHOICE
-------------------------
The routing is ordinary Python, not a prompt. The division of labour is:

    The LLM answers  "does this transcript window mean what was asked for?"
                     - a semantic judgement, which is what it is good at.

    This file answers "given that answer, what should happen next?"
                     - a control-flow decision, which needs to be reproducible,
                       testable, free to run, and defensible line by line.

Asking a model to decide "retry, skip or escalate?" would produce a decision I
could not reproduce, could not unit test, and could only explain by quoting the
model's own account of itself. `route()` below is a pure function of its inputs:
the same signals always produce the same decision and the same sentence
explaining it.

THE LADDER
----------
A request that does not match is not immediately given up on. It escalates
through strategies that attack different reasons for failing:

    1. baseline      top-5 windows, strict verification
    2. expand        the model rewrites the request into other phrasings, in
                     case the speaker used different words than the user did
    3. widen         top-15 plus a literal word-overlap pass, for names,
                     acronyms and numbers that embeddings blur together
    4. retranscribe  if - and only if - the transcript has holes, re-read those
                     specific chunks and try again

Then a terminal decision: `unfound` if the whole video was searched,
`escalated` if part of it could not be.

THE GUARD I CARE MOST ABOUT
---------------------------
Strategy 4 checks coverage first. If the transcript is complete, re-transcribing
cannot surface content that was already fully read, so the agent declines to
spend money on it and logs why. Knowing which actions are pointless is worth
more than being willing to try everything.
"""

from config import (
    FULL_COVERAGE_RATIO,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_STRATEGIES_PER_REQUEST,
    MIN_CLIP_SECONDS,
    NARROW_TO_UTTERANCES,
    TOP_K_BASELINE,
    TOP_K_WIDE,
)
from clipper import cutClip
from database import (
    getCoverage,
    getFailedSegments,
    logDecision,
    saveRequestOutcome,
    transcriptVersion,
)
from matcher import (
    ensureSearchIndex,
    expandQuery,
    lexicalCandidates,
    mergeCandidates,
    narrowToUtterances,
    retrieveCandidates,
    retrieveExpanded,
    scoreWindows,
    verifyCandidates,
)

STRATEGY_ORDER = ["baseline", "expand", "widen", "retranscribe"]

TERMINAL_STATUSES = {"fulfilled", "unfound", "escalated", "failed"}


# --------------------------------------------------------------------------
# should this request be worked at all?
# --------------------------------------------------------------------------

def shouldProcess(request, current_transcript_version: str):
    """
    Decide whether a request needs work this run. Returns (process, reason).

    The interesting case is a request previously answered "not found".
    Re-asking costs embedding and LLM calls, and if the transcript is exactly
    what it was when that conclusion was reached, the answer is guaranteed to
    be identical. The question is not "did this fail?" but "has the evidence
    changed since I decided?".
    """
    status = request["status"]

    if status not in TERMINAL_STATUSES:
        return True, "not yet attempted"

    if status == "fulfilled":
        return False, "already fulfilled - the clip exists"

    if status == "failed":
        # A failed cut is an infrastructure problem - disk, ffmpeg,
        # permissions - not a conclusion about the video's contents.
        return True, "previous attempt failed while cutting - retrying"

    if request["transcript_version"] == current_transcript_version:
        return False, (
            f"already {status} and the transcript is unchanged - re-asking "
            f"would reach the same conclusion at the same cost"
        )

    return True, (
        f"previously {status}, but the transcript has changed since "
        f"(a chunk that was unreadable is now transcribed) - reconsidering"
    )


# --------------------------------------------------------------------------
# the router - a pure function, so it can be tested and argued with
# --------------------------------------------------------------------------

def route(matched_window, confidence, strategies_left, coverage_complete):
    """
    Given the outcome of one strategy, decide what happens next.

    Returns (decision, reasoning). Decisions:

        accept          confident match - cut it
        hold            a match, but weak. Keep it as a fallback and try a
                        better strategy first rather than settling immediately
        next_strategy   no match, but there is something else worth trying
        exhausted       no match and nothing left to try

    Deliberately takes plain values rather than database rows, so it can be
    called in a test without a database, a video, or an API key.
    """
    if matched_window is not None:
        if confidence >= LOW_CONFIDENCE_THRESHOLD:
            return "accept", (
                f"Confidence {confidence:.2f} is at or above the "
                f"{LOW_CONFIDENCE_THRESHOLD} threshold - accepting this match."
            )

        if strategies_left:
            return "hold", (
                f"Confidence {confidence:.2f} is below the "
                f"{LOW_CONFIDENCE_THRESHOLD} threshold. Keeping this as a "
                f"fallback but trying a stronger strategy before settling - a "
                f"weak match is worth beating, not worth discarding."
            )

        return "accept", (
            f"Confidence {confidence:.2f} is below the "
            f"{LOW_CONFIDENCE_THRESHOLD} threshold, but no strategies remain. "
            f"Accepting and flagging for human review rather than reporting "
            f"nothing when something was found."
        )

    if strategies_left:
        return "next_strategy", (
            f"No candidate matched. {len(strategies_left)} strategy/strategies "
            f"remain ({', '.join(strategies_left)}) - escalating rather than "
            f"concluding the content is absent."
        )

    return "exhausted", (
        "No candidate matched and every strategy has been tried. "
        + ("Coverage is complete, so the video was fully searched."
           if coverage_complete else
           "Coverage is incomplete, so part of the video was never searched.")
    )


# --------------------------------------------------------------------------
# running one strategy
# --------------------------------------------------------------------------

def runStrategy(strategy: str, query: str, request_id: str, attempt: int,
                retranscribe_fn=None):
    """
    Execute one retrieval strategy. Returns (candidates, detail).

    `detail` is extra evidence for the log - the phrasings expansion produced,
    the words the lexical pass hit on, the chunks re-transcription repaired.
    """
    if strategy == "baseline":
        return retrieveCandidates(query, TOP_K_BASELINE), {}

    if strategy == "expand":
        phrasings = expandQuery(query)
        candidates = retrieveExpanded([query] + phrasings, TOP_K_BASELINE)
        return candidates, {"phrasings": phrasings}

    if strategy == "widen":
        semantic = scoreWindows([query])[:TOP_K_WIDE]
        lexical = lexicalCandidates(query, TOP_K_WIDE)
        candidates = mergeCandidates(semantic, lexical, limit=TOP_K_WIDE)
        return candidates, {
            "lexical_hits": [
                {"window_id": c["window_id"], "words": c.get("matched_words", [])}
                for c in lexical[:3]
            ]
        }

    if strategy == "retranscribe":
        repaired = []
        failed = []

        for segment in getFailedSegments():
            ok = retranscribe_fn(segment, query)
            (repaired if ok else failed).append(segment["chunk_name"])

        if repaired:
            # The transcript changed, so the search index is stale. This is the
            # rebuild that the original "if windows exist, do nothing" check
            # made impossible.
            ensureSearchIndex(force=True)

        candidates = retrieveCandidates(query, TOP_K_BASELINE)
        return candidates, {"repaired": repaired, "still_failed": failed}

    raise ValueError(f"Unknown strategy: {strategy}")


def validateWindow(verification, candidates):
    """
    Resolve the model's chosen window id against what it was actually shown.

    Structured output guarantees the SHAPE of an answer, not its truth. A model
    can return a perfectly well-formed integer naming a window it never saw,
    and cutting on it would produce a clip from an arbitrary point in the video
    with no error anywhere to explain it.
    """
    if not verification.matched or verification.window_id is None:
        return None, None

    window = next(
        (c for c in candidates if c["window_id"] == verification.window_id),
        None,
    )

    if window is None:
        return None, (
            f"Model returned window {verification.window_id}, which was not "
            f"among the candidates it was shown "
            f"({[c['window_id'] for c in candidates]}). Treating as no match "
            f"rather than cutting from an unknown span."
        )

    return window, None


def collectSignals(verification, candidates, coverage, detail=None):
    signals = {
        "best_similarity": round(candidates[0]["similarity"], 3) if candidates else 0.0,
        "candidates_seen": len(candidates),
        "llm_matched": bool(verification.matched),
        "llm_confidence": round(float(verification.confidence), 3),
        "llm_window_id": verification.window_id,
        "coverage_ratio": round(coverage["ratio"], 3),
        "gaps": [
            f"{g['start']:.0f}-{g['end']:.0f}s ({g['chunk_name']})"
            for g in coverage["gaps"]
        ],
    }

    if detail:
        signals.update(detail)

    return signals


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def processRequest(request, retranscribe_fn=None):
    """Take one request from wherever it is to a terminal state."""
    request_id = request["request_id"]
    query = request["query"]
    version_at_start = transcriptVersion()

    strategies_tried = []
    held_match = None          # a weak match kept while we try to beat it
    attempt = 0

    plan = STRATEGY_ORDER[:MAX_STRATEGIES_PER_REQUEST]

    for index, strategy in enumerate(plan):
        coverage = getCoverage()
        strategies_left = plan[index + 1:]

        # --- the guard: is this strategy even capable of helping? ----------
        if strategy == "retranscribe":
            if coverage["ratio"] >= FULL_COVERAGE_RATIO:
                logDecision(
                    stage="match", request_id=request_id, attempt=attempt,
                    strategy="retranscribe",
                    signals={"coverage_ratio": round(coverage["ratio"], 3),
                             "gap_count": 0},
                    decision="decline",
                    reasoning=(
                        "Declined to re-transcribe: the transcript already "
                        "covers 100% of the video, so re-reading audio cannot "
                        "surface content that was missed. Spending money here "
                        "could not change the outcome."
                    ),
                )
                continue

            if retranscribe_fn is None:
                logDecision(
                    stage="match", request_id=request_id, attempt=attempt,
                    strategy="retranscribe",
                    signals={"coverage_ratio": round(coverage["ratio"], 3)},
                    decision="decline",
                    reasoning="Declined: no re-transcription handler available.",
                )
                continue

        attempt += 1
        strategies_tried.append(strategy)

        candidates, detail = runStrategy(
            strategy, query, request_id, attempt, retranscribe_fn
        )

        verification = verifyCandidates(query, candidates)
        coverage = getCoverage()           # retranscribe may have changed it
        matched_window, invalid_reason = validateWindow(verification, candidates)
        signals = collectSignals(verification, candidates, coverage, detail)

        if invalid_reason:
            logDecision(
                stage="match", request_id=request_id, attempt=attempt,
                strategy=strategy, signals=signals,
                decision="reject_invalid_window", reasoning=invalid_reason,
            )

        decision, reasoning = route(
            matched_window=matched_window,
            confidence=float(verification.confidence),
            strategies_left=strategies_left,
            coverage_complete=coverage["ratio"] >= FULL_COVERAGE_RATIO,
        )

        logDecision(
            stage="match", request_id=request_id, attempt=attempt,
            strategy=strategy, signals=signals,
            decision=decision,
            reasoning=f"{verification.reason} {reasoning}".strip(),
        )

        if decision == "accept":
            return _fulfil(request_id, query, attempt, version_at_start,
                           strategies_tried, matched_window, verification,
                           signals, low_confidence=(
                               float(verification.confidence)
                               < LOW_CONFIDENCE_THRESHOLD))

        if decision == "hold":
            held_match = (matched_window, verification, signals)

    # --- ladder exhausted ---------------------------------------------------
    if held_match is not None:
        matched_window, verification, signals = held_match
        logDecision(
            stage="match", request_id=request_id, attempt=attempt,
            strategy="fallback", signals=signals, decision="accept_held",
            reasoning=(
                f"No stronger strategy beat the earlier weak match "
                f"(confidence {verification.confidence:.2f}). Accepting it and "
                f"flagging for review rather than reporting nothing."
            ),
        )
        return _fulfil(request_id, query, attempt, version_at_start,
                       strategies_tried, matched_window, verification, signals,
                       low_confidence=True)

    return _giveUp(request_id, attempt, version_at_start, strategies_tried,
                   getCoverage())


def enforceMinimumLength(start_time: float, end_time: float):
    """
    Grow a very short selection around its midpoint.

    A single sentence that answers the request can be two seconds long. That is
    technically the precise answer and useless to watch, so anything under
    MIN_CLIP_SECONDS is expanded symmetrically. cutClip clamps the result to
    the real bounds of the video, so growing past the end is safe.
    """
    duration = end_time - start_time

    if duration >= MIN_CLIP_SECONDS:
        return start_time, end_time, False

    shortfall = (MIN_CLIP_SECONDS - duration) / 2.0
    return max(0.0, start_time - shortfall), end_time + shortfall, True


def selectSpan(request_id, query, attempt, matched_window, signals):
    """
    Choose what to actually cut: the precise utterances, or the whole window.

    Returns (start_time, end_time).
    """
    window_start = matched_window["start_time"]
    window_end = matched_window["end_time"]

    if not NARROW_TO_UTTERANCES:
        return window_start, window_end

    start_time, end_time, reason = narrowToUtterances(query, matched_window)

    if start_time is None:
        # Narrowing is a precision improvement, not a requirement. When it
        # cannot run, fall back to the behaviour that existed before it - a
        # coarser clip, never a failed one - and record why.
        logDecision(
            stage="clip", request_id=request_id, attempt=attempt,
            strategy="narrow", signals=signals, decision="narrow_fallback",
            reasoning=f"{reason} Cutting the full window "
                      f"{window_start:.1f}-{window_end:.1f}s instead.",
        )
        return window_start, window_end

    start_time, end_time, widened = enforceMinimumLength(start_time, end_time)

    logDecision(
        stage="clip", request_id=request_id, attempt=attempt,
        strategy="narrow", decision="narrowed",
        signals={
            **signals,
            "window_span": round(window_end - window_start, 1),
            "narrowed_span": round(end_time - start_time, 1),
            "widened_to_minimum": widened,
        },
        reasoning=(
            f"Narrowed the {window_end - window_start:.0f}s window to "
            f"{start_time:.1f}-{end_time:.1f}s "
            f"({end_time - start_time:.0f}s). {reason}"
            + (f" Expanded to the {MIN_CLIP_SECONDS:.0f}s minimum clip length."
               if widened else "")
        ),
    )

    return start_time, end_time


def _fulfil(request_id, query, attempt, version, strategies_tried,
            matched_window, verification, signals, low_confidence=False):
    start_time, end_time = selectSpan(
        request_id, query, attempt, matched_window, signals
    )

    clip_result = cutClip(start_time=start_time, end_time=end_time)

    if clip_result["status"] == "failed":
        logDecision(
            stage="clip", request_id=request_id, attempt=attempt,
            strategy="cut",
            signals={**signals, "clip_error": clip_result["error"]},
            decision="fail",
            reasoning=(
                f"Window {matched_window['window_id']} matched, but cutting it "
                f"failed: {clip_result['error']}"
            ),
        )

        saveRequestOutcome(
            request_id=request_id, status="failed",
            resolution=f"Matched, but the clip could not be cut: {clip_result['error']}",
            window_id=matched_window["window_id"],
            confidence=verification.confidence,
            similarity=matched_window["similarity"],
            strategies_tried=strategies_tried, attempts=attempt,
            transcript_version=version,
        )

        return {"request_id": request_id, "status": "failed",
                "clip": None, "resolution": clip_result["error"],
                "strategies": strategies_tried}

    flag = " [LOW CONFIDENCE - review]" if low_confidence else ""

    resolution = (
        f"Found by '{strategies_tried[-1]}' in window "
        f"{matched_window['window_id']} "
        f"({matched_window['start_time']:.1f}-{matched_window['end_time']:.1f}s), "
        f"confidence {verification.confidence:.2f}. "
        f"Clip {clip_result['status']}.{flag}"
    )

    logDecision(
        stage="clip", request_id=request_id, attempt=attempt, strategy="cut",
        signals={**signals, "clip_status": clip_result["status"]},
        decision="clip_" + clip_result["status"],
        reasoning=(
            f"Cut {clip_result['start']:.1f}-{clip_result['end']:.1f}s to "
            f"{clip_result['path'].name} ({clip_result['status']})."
        ),
    )

    saveRequestOutcome(
        request_id=request_id,
        status="fulfilled", resolution=resolution,
        window_id=matched_window["window_id"],
        start_time=clip_result["start"], end_time=clip_result["end"],
        clip_path=clip_result["path"], confidence=verification.confidence,
        similarity=matched_window["similarity"],
        strategies_tried=strategies_tried, attempts=attempt,
        transcript_version=version,
    )

    return {"request_id": request_id, "status": "fulfilled",
            "clip": clip_result["path"], "resolution": resolution,
            "strategies": strategies_tried}


def _giveUp(request_id, attempt, version, strategies_tried, coverage):
    """
    Nothing matched. Decide what KIND of "no" this is.

    This distinction is the one that separates an agent from a search box:

        unfound   - the transcript is complete and this content is not in it.
                    A confident negative, and a real answer.

        escalated - part of the audio could not be read, and the answer may be
                    sitting in the part we cannot see. Reporting that as "not
                    found" would be a lie by omission.
    """
    tried = ", ".join(strategies_tried) or "none"
    complete = coverage["ratio"] >= FULL_COVERAGE_RATIO

    if complete:
        reasoning = (
            f"Tried {len(strategies_tried)} strategies ({tried}) with no match. "
            f"Transcript covers {coverage['ratio'] * 100:.1f}% of the video "
            f"with no gaps, so the whole recording was searched: this content "
            f"is not in the video."
        )
        status = "unfound"
        decision = "skip"
    else:
        gap_text = ", ".join(
            f"{g['start']:.0f}-{g['end']:.0f}s ({g['chunk_name']})"
            for g in coverage["gaps"]
        )
        reasoning = (
            f"Tried {len(strategies_tried)} strategies ({tried}) with no match, "
            f"and re-transcription could not repair every gap. Transcript "
            f"covers only {coverage['ratio'] * 100:.1f}% of the video - "
            f"unreadable: {gap_text}. The content cannot be ruled out, because "
            f"part of the recording was never searched. Needs a human."
        )
        status = "escalated"
        decision = "escalate"

    logDecision(
        stage="match", request_id=request_id, attempt=attempt,
        strategy="terminal",
        signals={"coverage_ratio": round(coverage["ratio"], 3),
                 "strategies_tried": strategies_tried},
        decision=decision, reasoning=reasoning,
    )

    saveRequestOutcome(
        request_id=request_id, status=status, resolution=reasoning,
        strategies_tried=strategies_tried, attempts=attempt,
        transcript_version=version,
    )

    return {"request_id": request_id, "status": status, "clip": None,
            "resolution": reasoning, "strategies": strategies_tried}
