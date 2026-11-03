"""Gemini Batch REST transport: file upload, batch create/get, result download.
One HTTP exchange per method; the driver holds all policy. Injectable for tests."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from .. import config
from ..stages import read as reader

API = "https://generativelanguage.googleapis.com"
JSONL_MIME = "application/jsonl"

# Batch job states, as the REST metadata spells them.
TERMINAL = {"BATCH_STATE_SUCCEEDED", "BATCH_STATE_FAILED", "BATCH_STATE_CANCELLED",
            "BATCH_STATE_EXPIRED"}
SUCCEEDED = "BATCH_STATE_SUCCEEDED"


class GeminiBatchApi:
    def __init__(self, api: str = API, key: str | None = None, timeout_s: int | None = None):
        self._api = api.rstrip("/")
        self._key = key
        self._timeout = timeout_s or config.READER_TIMEOUT_S

    # --- plumbing ----------------------------------------------------------

    def _headers(self, extra: dict | None = None) -> dict:
        h = {"x-goog-api-key": self._key or reader._key()}
        h.update(extra or {})
        return h

    def _json(self, method: str, url: str, body: dict | None = None, headers: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        hdrs = self._headers({"Content-Type": "application/json"} if body is not None else None)
        hdrs.update(headers or {})
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            return json.load(r)

    # --- surface -----------------------------------------------------------

    def upload_jsonl(self, path: Path, display_name: str) -> str:
        """Resumable upload of one JSONL file -> its Files API name
        (files/...). Streams the file body; nothing is read into memory
        beyond urllib's buffering."""
        size = path.stat().st_size
        start = urllib.request.Request(
            f"{self._api}/upload/v1beta/files",
            data=json.dumps({"file": {"display_name": display_name}}).encode(),
            headers=self._headers({
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": JSONL_MIME,
                "Content-Type": "application/json",
            }),
            method="POST",
        )
        with urllib.request.urlopen(start, timeout=self._timeout) as r:
            upload_url = r.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise RuntimeError("Files API returned no X-Goog-Upload-URL")
        with path.open("rb") as fh:
            finish = urllib.request.Request(
                upload_url, data=fh,
                headers={
                    "Content-Length": str(size),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
                method="POST",
            )
            with urllib.request.urlopen(finish, timeout=self._timeout) as r:
                info = json.load(r)
        return info["file"]["name"]

    def create_batch(self, model: str, file_name: str, display_name: str) -> dict:
        """-> the long-running operation: {"name": "batches/...", "metadata": {...}}."""
        return self._json(
            "POST", f"{self._api}/v1beta/models/{model}:batchGenerateContent",
            {"batch": {"display_name": display_name, "input_config": {"file_name": file_name}}},
        )

    def get_batch(self, name: str) -> dict:
        return self._json("GET", f"{self._api}/v1beta/{name}")

    def download_lines(self, file_name: str) -> Iterator[dict]:
        """The responses file, one parsed JSON object per line."""
        req = urllib.request.Request(
            f"{self._api}/download/v1beta/{file_name}:download?alt=media",
            headers=self._headers(),
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            for raw in r:
                raw = raw.strip()
                if raw:
                    yield json.loads(raw)


def batch_state(op: dict) -> str:
    """State string out of a get_batch / create_batch response."""
    meta = op.get("metadata") or {}
    return meta.get("state") or ("BATCH_STATE_SUCCEEDED" if op.get("done") else "BATCH_STATE_PENDING")


def responses_file(op: dict) -> str | None:
    """Where the results landed, whichever of the documented spots the
    server used (response.responsesFile, or metadata.output.responsesFile;
    snake_case tolerated)."""
    for container in (op.get("response") or {}, (op.get("metadata") or {}).get("output") or {}):
        for k in ("responsesFile", "responses_file"):
            if container.get(k):
                return container[k]
    return None
