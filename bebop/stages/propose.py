"""MinerU layout adapter: run MinerU, map middle.json para_blocks onto
LayoutElement. Regions and order only, never table content. Raw middle.json
is cached by (document sha256, adapter version) and re-flattened on load."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

from .. import config
from ..model import LayoutElement

# Per-host MinerU binary; absolute on provisioned hosts.
MINERU_BIN = os.environ.get("MINERU_BIN", str(config._SERVICE_ROOT / ".venv/bin/mineru"))
# When set (a running mineru-vllm-server), every call is a thin client of
# that one persistent engine instead of booting a vlm-engine per document.
_SERVER_URL = os.environ.get("MINERU_SERVER_URL", "")
MINERU_ARGS = (
    ("-b", "vlm-http-client", "-u", _SERVER_URL)
    if _SERVER_URL
    else ("-b", "vlm-engine", "--gpu-memory-utilization", "0.42")
)
MINERU_TIMEOUT_S = 1800

# Cache key half: bump when the CLI/model changes behavior. Derived from
# `mineru --version` at first use so an upgrade can never replay stale
# proposals silently.
_version_cache: str | None = None


def propose(pdf_path: str | Path) -> list[LayoutElement]:
    """Whole-document proposals, sorted by (page, reading_order)."""
    middle = _middle_json(Path(pdf_path))
    elements: list[LayoutElement] = []
    for page in middle["pdf_info"]:
        pno = page["page_idx"] + 1  # 1-based
        elements.extend(_page_elements(page.get("para_blocks", []), pno))
    return elements


class _Raw(NamedTuple):
    key: float  # MinerU's index, the sort key (absent -> last)
    type: str
    bbox: tuple
    header_rows: int | None = None
    # caption/footnote parts: position of the owning entry — relative to
    # _flatten_block's output, absolute once _page_elements places it
    owner: int | None = None


def _page_elements(para_blocks: list[dict], pno: int) -> list[LayoutElement]:
    """One page's para_blocks -> elements in dense reading order. A caption
    or footnote part keeps MinerU's attachment as `owner`: its table/image
    element's reading_order on this page."""
    raw: list[_Raw] = []
    for blk in para_blocks:
        base = len(raw)
        raw.extend(
            r if r.owner is None else r._replace(owner=base + r.owner)
            for r in _flatten_block(blk)
        )
    # MinerU's index is column-aware but sparse and occasionally absent;
    # sort by it (absent -> last, stable) and re-number dense 0-based.
    order = sorted(range(len(raw)), key=lambda i: raw[i].key)
    dense = {i: n for n, i in enumerate(order)}
    return [
        LayoutElement(
            type=raw[i].type, page=pno, bbox=tuple(raw[i].bbox),
            reading_order=n, header_rows=raw[i].header_rows,
            owner=None if raw[i].owner is None else dense[raw[i].owner],
        )
        for n, i in enumerate(order)
    ]


# MinerU block type -> element type. Unknown categories degrade to "text":
# an unrecognized region must reach the prose lane, never be dropped (the
# gate can only rescue words NO element covers).
_TYPE_MAP = {
    "table": "table",
    "image": "image",
    # a chart is a rendered graphic: it belongs in the images inventory.
    # Its axis labels/title are text-layer words the element claims — they
    # ship as prose behind the image block (emit), never silenced.
    "chart": "image",
    "list": "list",
    "text": "text",
    # MinerU's own heading call, kept as a role (#700) — emit writes it as
    # a "heading" block; still prose, so the gate treats it like text.
    "title": "title",
}


def _part_role(ptype: str) -> str | None:
    """A table/image/chart part MinerU marked as a caption or footnote."""
    if ptype.endswith("_caption"):
        return "caption"
    if ptype.endswith("_footnote"):
        return "footnote"
    return None


def _flatten_block(blk: dict) -> list[_Raw]:
    btype = blk.get("type")
    index = blk.get("index")
    key = float(index) if isinstance(index, (int, float)) else float(1 << 30)
    if btype == "list":
        out = []
        for item in blk.get("blocks", []) or [blk]:
            iidx = item.get("index")
            ikey = float(iidx) if isinstance(iidx, (int, float)) else key
            out.append(_Raw(ikey, "list", tuple(item["bbox"])))
        return out
    if btype in ("table", "image", "chart"):
        header_rows = _table_header_rows(blk) if btype == "table" else None
        out = [_Raw(key, _TYPE_MAP[btype], tuple(blk["bbox"]), header_rows)]
        for part in blk.get("blocks", []):
            role = _part_role(part.get("type", ""))
            if role is not None:
                pidx = part.get("index")
                pkey = float(pidx) if isinstance(pidx, (int, float)) else key
                # owner=0: the composite's own entry, first in `out`
                out.append(_Raw(pkey, role, tuple(part["bbox"]), owner=0))
        return out
    return [_Raw(key, _TYPE_MAP.get(btype, "text"), tuple(blk["bbox"]))]


def _table_header_rows(blk: dict) -> int | None:
    """Header row count from MinerU's table HTML: <thead> rows, else the run
    of leading all-<th> rows, else None."""
    html = ""
    for part in blk.get("blocks", []):
        if part.get("type") == "table_body":
            for line in part.get("lines", []):
                for span in line.get("spans", []):
                    html = span.get("html") or html
    if not html:
        return None
    m = re.search(r"<thead.*?</thead>", html, re.S | re.I)
    if m:
        return len(re.findall(r"<tr[\s>]", m.group(0), re.I)) or None
    count = 0
    for row in re.findall(r"<tr[\s>].*?</tr>", html, re.S | re.I):
        cells = re.findall(r"<(t[dh])[\s>]", row, re.I)
        if cells and all(c.lower() == "th" for c in cells):
            count += 1
        else:
            break
    return count or None


def _middle_json(pdf_path: Path) -> dict:
    cached = _cache_path(pdf_path)
    if cached.exists():
        return json.loads(cached.read_text())
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [str(Path.cwd() / MINERU_BIN), "-p", str(pdf_path), "-o", td, *MINERU_ARGS],
            capture_output=True,
            timeout=MINERU_TIMEOUT_S,
            check=True,
        )
        mids = list(Path(td).glob("*/vlm/*_middle.json"))
        if not mids:
            raise FileNotFoundError(
                f"MinerU produced no middle.json for {pdf_path.name} — "
                f"see its stderr in the CalledProcessError above, or rerun by hand"
            )
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(mids[0], cached)
    return json.loads(cached.read_text())


def _cache_path(pdf_path: Path) -> Path:
    sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return config._SERVICE_ROOT / config.LAYOUT_CACHE_DIR / f"{sha}-{_version()}" / "middle.json"


def _version() -> str:
    global _version_cache
    if _version_cache is None:
        try:
            out = subprocess.run(
                [str(Path.cwd() / MINERU_BIN), "--version"],
                capture_output=True, text=True, timeout=60,
            ).stdout
            _version_cache = "mineru-" + (out.split()[-1].strip() or "unknown")
        except Exception:
            _version_cache = "mineru-unknown"
    return _version_cache
