"""Document in, assets out: ingest -> propose -> cover -> read_table per
table (parallel) -> compose -> emit. A failed read degrades to prose."""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .stages import compose
from .stages.emit import emit
from .stages.cover import cover
from .stages.propose import propose
from .model import Document, Grid
from .stages.read import _document, read_table


@dataclass
class ParseResult:
    document: Document
    assets: dict
    failures: list[dict] = field(default_factory=list)  # per-element read failures
    flags: list[dict] = field(default_factory=list)  # verification findings
    telemetry: dict = field(default_factory=dict)


@dataclass
class Layout:
    """Everything parse() knows before the first table read. Its own step so
    the batch pre-warm enumerates exactly the tables a parse would read."""

    document: Document
    covers: dict
    merge_audit: list[dict]
    proposals: int
    timings: dict[str, float]

    @property
    def table_elements(self) -> list:
        """The elements the table lane reads, in cover order."""
        return [
            e for cov in self.covers.values() for e in cov.elements if e.type == "table"
        ]



def _unanchored_numerics(grid: Grid) -> int:
    """Cells that attempted anchoring, carry a digit, and claimed nothing."""
    return sum(
        1
        for row in grid.rows
        for c in row.cells
        if c.total > 0 and not c.claims and any(ch.isdigit() for ch in c.text)
    )


def _better(first: Grid, second: Grid) -> Grid:
    """Fewer page-denied numerics wins; then more claimed words; ties keep first."""
    u1, u2 = _unanchored_numerics(first), _unanchored_numerics(second)
    if u2 < u1:
        return second
    claimed = lambda g: sum(len(c.claims) for row in g.rows for c in row.cells)  # noqa: E731
    if u2 == u1 and claimed(second) > claimed(first):
        return second
    return first

def lay_out(pdf_path: str | Path) -> Layout:
    """ingest → propose (MinerU, cached) → band merge → cover, per page."""
    timings: dict[str, float] = {}

    t = time.time()
    # Through the reader's per-path cache, not ingest directly: read_table
    # re-ingests behind its stated seam, so priming the cache here keeps
    # the pool's threads from stampeding duplicate ingests of a cold one.
    doc = _document(str(pdf_path))
    timings["ingest"] = round(time.time() - t, 2)

    t = time.time()
    elements = propose(pdf_path)
    timings["layout"] = round(time.time() - t, 2)

    t = time.time()
    # Band merge BEFORE the gate (the one readmitted composition rule):
    # footprint-aligned stacked table elements become one region, so the
    # header band reads with its values; every merge is audit-flagged.
    covers = {}
    merge_audit: list[dict] = []
    for p in doc.pages:
        page_els = compose.merge_bands(
            p, [e for e in elements if e.page == p.number], audit=merge_audit
        )
        covers[p.number] = cover(p, page_els)
    timings["gate"] = round(time.time() - t, 2)
    return Layout(
        document=doc, covers=covers, merge_audit=merge_audit,
        proposals=len(elements), timings=timings,
    )


def parse(pdf_path: str | Path, transport=None) -> ParseResult:
    layout = lay_out(pdf_path)
    doc, covers, merge_audit = layout.document, layout.covers, layout.merge_audit
    timings = layout.timings

    t = time.time()
    table_els = layout.table_elements
    grids: list[Grid] = []
    failures: list[dict] = []

    rereads = 0

    def _read(el):
        nonlocal rereads
        try:
            grid = read_table(pdf_path, el, transport=transport)
            # A digit-bearing cell the page denies buys one fresh read; keep the better.
            if _unanchored_numerics(grid):
                rereads += 1
                try:
                    retry = read_table(pdf_path, el, transport=transport, variant="retry")
                except Exception:  # noqa: BLE001 — a failed retry never costs the read we have
                    retry = None
                if retry is not None:
                    grid = _better(grid, retry)
            return el, grid, None
        except Exception as exc:  # noqa: BLE001 — a failed read degrades, never raises
            return el, None, str(exc)[:200]

    with ThreadPoolExecutor(max_workers=config.READER_PARALLELISM) as ex:
        for el, grid, err in ex.map(_read, table_els):
            if grid is not None:
                grids.append(grid)
            else:
                failures.append(
                    {"page": el.page, "reading_order": el.reading_order, "error": err}
                )
    timings["read"] = round(time.time() - t, 2)

    t = time.time()
    # Cross-page composition after reading: join spreads, then inherit headers.
    page_map = {p.number: p for p in doc.pages}
    compose_audit: list[dict] = []
    grids = compose.join_spreads(grids, page_map, audit=compose_audit)
    compose.inherit_headers(grids, page_map, audit=compose_audit)
    timings["compose"] = round(time.time() - t, 2)

    flags = merge_audit + compose_audit + [f for g in grids for f in g.flags]

    t = time.time()
    assets = emit(doc, covers, grids)
    timings["emit"] = round(time.time() - t, 2)

    telemetry = {
        "pages": len(doc.pages),
        "elements": layout.proposals,
        "element_types": dict(
            Counter(e.type for cov in covers.values() for e in cov.elements)
        ),
        "tables_read": len(grids),
        "tables_degraded": len(failures),
        "tables_reread": rereads,
        "spreads_joined": sum(
            1 for f in compose_audit if f["kind"] == "spread_joined"
        ),
        "headers_inherited": sum(
            1 for f in compose_audit if f["kind"] == "header_inherited"
        ),
        "seconds": timings,
    }
    return ParseResult(
        document=doc, assets=assets, failures=failures, flags=flags, telemetry=telemetry
    )
