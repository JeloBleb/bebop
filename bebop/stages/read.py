"""Vision transcription of table regions: read_table(pdf_path, element) -> Grid.
Render the region, read it (cached raw response), anchor and verify, mark headers."""

import base64
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pymupdf
from prefect.blocks.system import Secret

from .. import config
from .cover import _center_in  # the one containment rule — never re-derive
from .ingest import ingest
from ..model import Document, Grid, LayoutElement, Page, Word
from .anchor import verify_element

# Hashed into every asset's provenance stamp and the read-cache key.
PROMPT = """This image is ONE table. Transcribe ALL of its text verbatim as JSON: {"header_rows": N, "rows": [[cell, cell, ...], ...]}

Rules:
- One array per LOGICAL table row, in reading order. Cells are the row's fields, left to right.
- When a cell's text wraps onto more printed lines, keep ALL of its text in that one cell (join the wrapped lines with a space) - never start a new row for a continuation line.
- "header_rows" is the number of LEADING rows that are column headers (0 if the table prints none). A section label or a data row is never a header.
- Copy each cell EXACTLY as printed (keep $ , % ( ) / - formatting). Use null for blank cell positions.
- Do NOT correct, normalize, infer, or complete any value, even if it looks wrong or breaks a pattern.
- Include everything: headers, section labels, sub-rows, totals, footnotes. Nothing may be skipped.
- Output ONLY the JSON object, MINIFIED on a single line - no pretty-printing, no spaces after commas."""


class TableUnread(RuntimeError):
    """No usable transcription; the caller degrades the element to prose."""


@dataclass(frozen=True)
class Read:
    rows: tuple | None  # transcribed rows, or None when nothing usable came back
    error: str | None  # last failure, None on a clean parse
    cost: float  # USD reported by the provider (0.0 on replay, and 0.0 when the
    # endpoint reports no cost — Google's reports token counts only; those
    # land in the cache entry, issue #95)
    cached: bool  # served from the read cache
    header_rows: int | None = None  # the model's count of leading header rows


def render(pdf_path: str | Path, element: LayoutElement) -> bytes | None:
    """element.bbox + legibility pad -> PNG bytes. None when the clip is
    empty. (v2's render, with element.bbox replacing the crop hull — the
    gate already snapped the bbox word-tight.)"""
    with pymupdf.open(pdf_path) as doc:
        pg = doc[element.page - 1]
        box = element.bbox
        pad = config.RENDER_PAD_PT
        clip = pymupdf.Rect(box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad) & pg.rect
        if clip.is_empty:
            return None
        return pg.get_pixmap(dpi=config.RENDER_DPI, clip=clip).tobytes("png")


def read_png(png: bytes, transport=None, variant: str = "") -> Read:
    """Transcribe one image, cache-through. transport: png -> (raw text, usage),
    injectable for tests. The cache stores the raw response; rows are re-parsed
    on every read. Content retries here; throttle retries live in _throttled."""
    path = _cache_path(png, variant)
    if path.exists():
        entry = json.loads(path.read_text())
        raw = entry["raw"]
        if raw is not None:
            return _finish(raw, None, cost=0.0, cached=True)
        # raw:null is an audit entry for a transport failure, never a hit.

    transport = transport or _http_transport
    text, error = None, "unattempted"
    cost, prompt_tokens, completion_tokens = 0.0, 0, 0
    for attempt in range(config.READER_RETRIES):
        if attempt:
            time.sleep(config.READER_RETRY_SLEEP_S)
        try:
            text, attempt_usage = _throttled(transport, png)
        except Exception as e:  # noqa: BLE001 — transport failures become the error record
            error = str(e)[:300]
            continue
        attempt_cost, attempt_pt, attempt_ct = _usage(attempt_usage)
        cost += attempt_cost
        prompt_tokens += attempt_pt
        completion_tokens += attempt_ct
        rows, _hr, error = _parse(text)
        if rows is not None:
            break

    _store(path, _sha(png), text, error, cost, prompt_tokens, completion_tokens)
    return _finish(text, error if text is None else None, cost=cost, cached=False)


def _store(
    path: Path, image_sha: str, raw: str | None, error: str | None,
    cost: float, prompt_tokens: int, completion_tokens: int,
) -> None:
    """Write one read-cache entry. The single writer of the entry schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": config.READER_MODEL,
                "prompt_sha256": _sha(PROMPT.encode()),
                "image_sha256": image_sha,
                "raw": raw,
                "error": error,
                "cost": cost,
                # Additive fields; older entries without them still load.
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )
    )


def _finish(raw: str | None, transport_error: str | None, cost: float, cached: bool) -> Read:
    """Raw response -> Read: whole-JSON parse or nothing. A response that
    never parses yields rows=None and the caller degrades the element to
    its text-layer prose — content preserved on the honest path."""
    if raw is None:
        return Read(rows=None, error=transport_error, cost=cost, cached=cached)
    rows, header_rows, error = _parse(raw)
    if rows is None:
        return Read(rows=None, error=error, cost=cost, cached=cached)
    return Read(rows=_freeze(rows), error=None, cost=cost, cached=cached,
                header_rows=header_rows)


def _freeze(rows: list | None) -> tuple | None:
    return None if rows is None else tuple(tuple(r) for r in rows)



def _parse(text: str) -> tuple[list | None, int | None, str | None]:
    """-> (rows, the model's header_rows count when valid, error)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, None, "no JSON object in response"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, None, f"unparseable JSON: {e}"
    rows = obj.get("rows")
    if not isinstance(rows, list):
        return None, None, "no rows list in response"
    hr = obj.get("header_rows")
    hr = hr if isinstance(hr, int) and hr >= 0 else None
    return [r if isinstance(r, list) else [r] for r in rows], hr, None


# Google's Gemini API host. Model ids there are bare (gemini-3.7-flash);
# OpenRouter routes on provider-prefixed ids (google/gemini-3.7-flash).
_GOOGLE_API_HOST = "generativelanguage.googleapis.com"


def _wire_model() -> str:
    """READER_MODEL as the endpoint spells it: provider prefix stripped for
    Google's own endpoint, unchanged otherwise."""
    model = config.READER_MODEL
    if _GOOGLE_API_HOST in config.READER_API_URL:
        return model.rpartition("/")[2]
    return model


def _http_transport(png: bytes) -> tuple[str, dict]:
    """POST one image+prompt to config.READER_API_URL. -> (raw text, usage dict);
    absent usage fields pass through as zeros."""
    body = {
        "model": _wire_model(),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.standard_b64encode(png).decode()},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "temperature": config.READER_TEMPERATURE,
        "max_tokens": config.READER_MAX_TOKENS,
    }
    req = urllib.request.Request(
        config.READER_API_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=config.READER_TIMEOUT_S) as r:
        resp = json.load(r)
    usage = resp.get("usage") or {}
    return resp["choices"][0]["message"]["content"], {
        "cost": float(usage.get("cost", 0) or 0),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
    }


def _usage(u) -> tuple[float, int, int]:
    """usage dict or bare cost number -> (cost, prompt_tokens, completion_tokens)."""
    if isinstance(u, dict):
        return (
            float(u.get("cost", 0) or 0),
            int(u.get("prompt_tokens", 0) or 0),
            int(u.get("completion_tokens", 0) or 0),
        )
    return float(u or 0), 0, 0


class _CoolOff:
    """Shared cool-off latch: one throttled thread advances resume_at and
    every transport call waits it out, so a single 429 pauses all threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._resume_at = 0.0  # monotonic timestamp; in the past = open

    def wait(self) -> None:
        while True:  # resume_at can advance while we sleep — re-check
            with self._lock:
                remaining = self._resume_at - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(remaining)

    def trip(self, delay_s: float) -> None:
        with self._lock:
            self._resume_at = max(self._resume_at, time.monotonic() + delay_s)


_cooloff = _CoolOff()


def _throttle_class(e: Exception) -> bool:
    """HTTP 429 or 5xx — the provider's window is closed; the failure
    says nothing about the image. Everything else (URLError, timeouts,
    unparseable JSON) keeps the plain READER_RETRIES path."""
    return isinstance(e, urllib.error.HTTPError) and (e.code == 429 or 500 <= e.code <= 599)


def _retry_after(e: urllib.error.HTTPError) -> int | None:
    """The response's Retry-After as integer seconds, else None (absent,
    unparseable, or an HTTPError constructed without headers)."""
    try:
        return int(e.headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        return None


def _throttled(transport, png: bytes) -> tuple[str, float]:
    """One transport call inside the throttle envelope: wait out the cool-off,
    call, and on HTTP 429/5xx back off (capped exponential, jittered,
    Retry-After honored) on its own budget. Exhausted: the last error propagates."""
    for k in range(config.READER_THROTTLE_RETRIES + 1):
        _cooloff.wait()
        try:
            return transport(png)
        except urllib.error.HTTPError as e:
            if not _throttle_class(e) or k == config.READER_THROTTLE_RETRIES:
                raise
            delay = min(config.READER_THROTTLE_CAP_S,
                        config.READER_THROTTLE_BASE_S * 2 ** k) * random.uniform(0.5, 1.5)
            retry_after = _retry_after(e)
            if retry_after is not None:
                delay = max(retry_after, delay)
            _cooloff.trip(delay)
    raise AssertionError("unreachable: the final attempt returns or raises")


def _cache_model() -> str:
    """Model id the read cache is keyed on. A bare Gemini id gains the google/
    prefix so both spellings share one cache dir; other ids pass through."""
    model = config.READER_MODEL
    if "/" not in model and model.startswith("gemini"):
        return "google/" + model
    return model


def _cache_path(png: bytes, variant: str = "") -> Path:
    """variant keys an independent cached sample of the same image (the
    escalation retry) — both reads stay auditable, and the deterministic
    winner never depends on cache clobbering."""
    return _cache_path_for_sha(_sha(png), variant)


def _cache_path_for_sha(image_sha: str, variant: str = "") -> Path:
    """Cache path from the image sha alone; _cache_path is this plus the hashing."""
    model = _cache_model()
    ctx = _sha(f"{model}\n{PROMPT}".encode())[:12]
    slug = model.replace("/", "_")
    name = image_sha + (f"-{variant}" if variant else "")
    return config._SERVICE_ROOT / config.READ_CACHE_DIR / f"{slug}-{ctx}" / f"{name}.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_key_cache: str | None = None


def _key() -> str:
    """Reader API key from its Prefect Secret block, process-cached. Block name
    from $READER_KEY_BLOCK, then legacy $OPENROUTER_KEY_BLOCK, then config."""
    global _key_cache
    if _key_cache is not None:
        return _key_cache
    block_name = (
        os.environ.get("READER_KEY_BLOCK")
        or os.environ.get("OPENROUTER_KEY_BLOCK")
        or config.READER_KEY_BLOCK
    )
    try:
        block = Secret.load(block_name)
    except Exception as e:  # noqa: BLE001 — surfaced as one clear config error
        raise RuntimeError(
            f"Reader key block {block_name!r} could not be loaded from "
            f"Prefect ({e}). Register it as a Prefect Secret block, or set "
            f"$READER_KEY_BLOCK to point at an existing one."
        ) from e
    _key_cache = block.get()
    return _key_cache


@lru_cache(maxsize=16)
def _document(pdf_path: str) -> Document:
    """ingest is pure — the same path yields the same Document, so caching
    on the path keeps the read_table seam (pdf_path + element) without
    threading a Document parameter through."""
    return ingest(pdf_path)


def _element_lines(page: Page, element: LayoutElement) -> list[list[tuple[int, Word]]]:
    """The page's Lines restricted to the element's words (gate's containment rule)."""
    out = []
    for ln in page.lines:
        members = [
            (gi, page.words[gi])
            for gi in ln.words
            if _center_in(page.words[gi], element.bbox)
        ]
        if members:
            out.append(members)
    return out


def read_table(
    pdf_path: str | Path, element: LayoutElement, transport=None,
    variant: str = "",
) -> Grid:
    """Render, transcribe, anchor and verify one table element. Raises
    TableUnread when no usable transcription came back."""
    page = _document(str(pdf_path)).pages[element.page - 1]
    lines = _element_lines(page, element)

    png = render(pdf_path, element)
    if png is None:
        raise TableUnread(f"empty clip for element {element.reading_order} on page {element.page}")
    read = read_png(png, transport=transport, variant=variant)
    if read.rows is None:
        raise TableUnread(f"unread element {element.reading_order} on page {element.page}: {read.error}")

    rows, unclaimed, flags = verify_element(lines, read.rows, page=element.page)
    header_count = element.header_rows if element.header_rows else read.header_rows
    _mark_headers(rows, header_count)
    return Grid(element=element, rows=rows, flags=flags, unclaimed=tuple(unclaimed))


def _mark_headers(rows, header_count: int | None) -> None:
    """Mark header rows from element.header_rows or the reader's own count.
    Neither given: no headers, no guess."""
    if header_count is None:
        return
    marked = 0
    for row in rows:
        if marked >= header_count:
            break
        if row.kind == "data":
            row.kind = "header"
            marked += 1
