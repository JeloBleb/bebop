"""The four commands: plan, submit, status, collect. Plain functions over
(State, transport); each is idempotent and safe to re-run."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path

from .. import config
from ..stages import read as reader
from . import gemini, request
from .crops import enumerate_crops, is_cached
from .state import FAILED, PENDING, SUBMITTED, UNPARSED, WRITTEN, CropRecord, Job, State

log = logging.getLogger("bebop.batch")

STATE_FILE = "state.json"
CHUNK_PREFIX = "requests-"
DEFAULT_CHUNK_BYTES = 1 << 30  # 1 GiB; the Files API caps a file at 2 GB


def load_or_fresh(state_dir: Path) -> State:
    path = state_dir / STATE_FILE
    if path.exists():
        st = State.load(path)
        st.check_current()
        return st
    return State.fresh()


def save(state: State, state_dir: Path) -> None:
    state.save(state_dir / STATE_FILE)


# --- plan --------------------------------------------------------------------


class _ChunkWriter:
    """Streams JSONL lines into requests-NNN.jsonl files, rolling to a
    new file at the byte limit. Continues numbering after existing
    chunks so a resumed plan never clobbers a submitted file."""

    def __init__(self, state_dir: Path, state: State, chunk_bytes: int):
        self._dir = state_dir
        self._limit = chunk_bytes
        self._n = 1 + max(
            (int(j.chunk[len(CHUNK_PREFIX):-len(".jsonl")]) for j in state.jobs), default=-1
        )
        self._fh = None
        self._job: Job | None = None
        self.jobs: list[Job] = []

    def _open(self) -> None:
        name = f"{CHUNK_PREFIX}{self._n:03d}.jsonl"
        self._n += 1
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fh = (self._dir / name).open("w")
        self._job = Job(chunk=name)
        self.jobs.append(self._job)

    def write(self, line: str) -> str:
        size = len(line) + 1
        if self._fh is None or (self._job.bytes + size > self._limit and self._job.requests):
            self.close()
            self._open()
        self._fh.write(line + "\n")
        self._job.requests += 1
        self._job.bytes += size
        return self._job.chunk

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def plan(
    pdfs: Iterable[str | Path], state: State, state_dir: Path,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict:
    """Enumerate crops, write the cold ones into chunk files, record them
    in the state. Returns the counts the CLI reports."""
    seen_this_run: set[str] = set()
    n_crops = n_cached = n_dup = n_known = n_new = 0
    writer = _ChunkWriter(state_dir, state, chunk_bytes)
    try:
        for crop in enumerate_crops(pdfs):
            n_crops += 1
            sha = crop.image_sha
            if sha in seen_this_run:
                n_dup += 1
                continue
            seen_this_run.add(sha)
            if crop.cached:
                n_cached += 1
                rec = state.crops.get(sha)
                if rec is not None and rec.status not in (WRITTEN,):
                    rec.status = WRITTEN  # something else primed it meanwhile
                continue
            rec = state.crops.get(sha)
            if rec is not None and rec.status in (PENDING, SUBMITTED):
                n_known += 1
                continue
            chunk = writer.write(request.jsonl_line(sha, crop.png))
            state.crops[sha] = CropRecord(
                pdf=crop.pdf, page=crop.page, reading_order=crop.reading_order,
                chunk=chunk, status=PENDING,
            )
            n_new += 1
    finally:
        writer.close()
    state.jobs.extend(writer.jobs)
    save(state, state_dir)
    return {
        "crops": n_crops,
        "already_cached": n_cached,
        "duplicate_in_run": n_dup,
        "already_planned": n_known,
        "new_requests": n_new,
        "new_chunks": [j.chunk for j in writer.jobs],
        "unsubmitted_requests": sum(j.requests for j in state.jobs if j.name is None),
        "estimated_cost_usd": request.estimated_cost_usd(
            sum(j.requests for j in state.jobs if j.name is None)
        ),
        "estimated_cost_basis": (
            f"{request.INTERACTIVE_USD_PER_TABLE * 100:.1f}c/table interactive "
            f"x {request.BATCH_DISCOUNT:.0%} batch"
        ),
    }


# --- submit ------------------------------------------------------------------


def submit(state: State, state_dir: Path, api, dry_run: bool = False) -> dict:
    """Upload each unsubmitted chunk and create its batch job. --dry-run
    does everything but the two API calls: the files are on disk, the
    counts and the cost estimate are reported, and nothing is billed."""
    if gemini.API.split("//")[1] not in config.READER_API_URL:
        log.warning(
            "READER_API_URL is not Google's endpoint (%s); the key block %s the "
            "batch will authenticate with must still be a Gemini API key",
            config.READER_API_URL, config.READER_KEY_BLOCK,
        )
    todo = [j for j in state.jobs if j.name is None and j.requests]
    submitted: list[str] = []
    for job in todo:
        if dry_run:
            log.info("dry-run: would submit %s (%d requests, %d bytes)",
                     job.chunk, job.requests, job.bytes)
            continue
        path = state_dir / job.chunk
        display = f"bebop-prewarm-{state_dir.name}-{job.chunk.removesuffix('.jsonl')}"
        if job.file_name is None:
            job.file_name = api.upload_jsonl(path, display)
            save(state, state_dir)  # an upload is paid-for state: keep it across a crash
        op = api.create_batch(request.wire_model(), job.file_name, display)
        job.name = op["name"]
        job.state = gemini.batch_state(op)
        for rec in state.crops.values():
            if rec.chunk == job.chunk and rec.status == PENDING:
                rec.status = SUBMITTED
        save(state, state_dir)
        submitted.append(job.name)
        log.info("submitted %s as %s (%d requests)", job.chunk, job.name, job.requests)
    n = sum(j.requests for j in todo)
    return {
        "dry_run": dry_run,
        "chunks": [j.chunk for j in todo],
        "requests": n,
        "estimated_cost_usd": request.estimated_cost_usd(n),
        "jobs": submitted,
        "model": request.wire_model(),
    }


# --- status ------------------------------------------------------------------


def status(state: State, state_dir: Path, api) -> dict:
    """Refresh every live job's state from the API; report all of them."""
    for job in state.jobs:
        if job.name and not job.collected and job.state not in gemini.TERMINAL:
            op = api.get_batch(job.name)
            job.state = gemini.batch_state(op)
            job.responses_file = gemini.responses_file(op) or job.responses_file
    save(state, state_dir)
    return {
        "jobs": [
            {"chunk": j.chunk, "name": j.name, "requests": j.requests,
             "state": j.state or ("unsubmitted" if j.name is None else None),
             "collected": j.collected}
            for j in state.jobs
        ],
        "crops": state.counts(),
        "cache_dir": state.cache_dir,
    }


# --- collect -----------------------------------------------------------------


def collect(state: State, state_dir: Path, api, wait_s: float = 0.0, poll_s: float = 60.0) -> dict:
    """Write every succeeded job's responses into the read cache. With
    wait_s > 0, poll until the jobs finish (or the wait expires)."""
    deadline = time.monotonic() + wait_s
    while True:
        status(state, state_dir, api)
        live = [j for j in state.jobs if j.name and j.state not in gemini.TERMINAL]
        if not live or time.monotonic() >= deadline:
            break
        log.info("%d job(s) still running; polling again in %.0fs", len(live), poll_s)
        time.sleep(poll_s)

    written = unparsed = failed = unknown = skipped = 0
    for job in state.jobs:
        if job.collected or job.state != gemini.SUCCEEDED:
            continue
        if not job.responses_file:
            op = api.get_batch(job.name)
            job.responses_file = gemini.responses_file(op)
            if not job.responses_file:
                log.warning("%s succeeded but reports no responses file", job.name)
                continue
        for line in api.download_lines(job.responses_file):
            key = line.get("key")
            rec = state.crops.get(key)
            if rec is None:
                unknown += 1
                continue
            outcome = _land(key, rec, line)
            if outcome == WRITTEN:
                written += 1
            elif outcome == UNPARSED:
                unparsed += 1
            elif outcome == FAILED:
                failed += 1
            else:
                skipped += 1
        job.collected = True
        save(state, state_dir)
    return {
        "written": written, "unparsed": unparsed, "failed": failed,
        "already_cached": skipped, "unknown_keys": unknown,
        "jobs_pending": [j.name for j in state.jobs if j.name and not j.collected],
        "crops": state.counts(),
    }


def _land(key: str, rec: CropRecord, line: dict) -> str:
    """One response line -> the read cache via the reader's writer. Only a
    parseable response is written; an unparseable one leaves the cache cold."""
    path = reader._cache_path_for_sha(key)
    if is_cached(path):
        rec.status = WRITTEN
        return "skipped"
    if line.get("error"):
        rec.status, rec.error = FAILED, str(line["error"])[:300]
        return FAILED
    text, usage = request.response_text(line.get("response") or {})
    if text is None:
        rec.status, rec.error = FAILED, "no candidate text in response"
        return FAILED
    rows, _hr, err = reader._parse(text)
    if rows is None:
        rec.status, rec.error = UNPARSED, err
        return UNPARSED
    cost, pt, ct = reader._usage(usage)
    reader._store(path, key, text, None, cost, pt, ct)
    rec.status, rec.error = WRITTEN, None
    return WRITTEN
