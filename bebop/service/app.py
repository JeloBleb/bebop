"""Parse service HTTP surface: POST /parse (multipart PDF -> assets + flags +
telemetry + provenance), GET /health, /version. Envelope matches the v2 contract."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from bebop import __version__, config
from bebop.pipeline import parse as run_pipeline

app = FastAPI(title="bebop", version=__version__)

# One parse at a time by default; the layout lane shares one GPU engine.
_parse_slots = asyncio.Semaphore(int(os.environ.get("BEBOP_PARSE_CONCURRENCY", "1")))


@app.get("/health")
def health():
    return {"ok": True}


#: Written by ``scripts/setup-bebop-service.sh`` into the staged service
#: directory: the git tree hash of the ``bebop`` tree it was staged from
#: (#450). Absent in a checkout that was not staged, which reports ``null``.
TREE_FILE = Path(__file__).resolve().parents[1] / "BEBOP_TREE"


def staged_tree() -> str | None:
    """The staged tree hash, or ``None`` when there is none to report."""
    try:
        value = TREE_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


@app.get("/version")
def version():
    # ``tree`` identifies the code, which ``service`` (a hand-kept string)
    # does not. Runs record it; nothing holds on it — one bebop serves
    # every tier's workers.
    return {
        "service": __version__,
        "reader": config.READER_MODEL,
        "tree": staged_tree(),
    }


@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    document_id: str | None = Form(default=None),
    pipeline_id: str | None = Form(default=None),
):
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(data)
        path = tf.name
    try:
        # Off the event loop: a parse is minutes of blocking work, and an
        # inline call would freeze /health (and every queued request) for
        # the duration.
        async with _parse_slots:
            result = await run_in_threadpool(run_pipeline, path)
    finally:
        os.unlink(path)
    flags = list(result.flags) + [
        {"page": f["page"], "kind": "table_unread", "severity": "high",
         "detail": f"table element {f['reading_order']}: {f['error']}"}
        for f in result.failures
    ]
    return {
        "assets": result.assets,
        "flags": flags,
        "queue": [],
        "verdicts": [],
        "telemetry": result.telemetry,
        "provenance": {
            "service": __version__,
            "reader": config.READER_MODEL,
            "document_hash": result.document.id,
            "document_id": document_id,
            "pipeline_id": pipeline_id,
        },
    }
