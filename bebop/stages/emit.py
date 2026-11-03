"""Emit the assets (tables.json, doc.json, images.json, markdown) from
(document, covers, grids). Pure function; every judgment was made upstream."""

import hashlib

from .. import __version__, config
from .cover import Cover, GateElement
from ..model import Cell, Document, Grid, LayoutElement, Page
from .read import PROMPT


def emit(doc: Document, covers: dict[int, Cover], grids: list[Grid]) -> dict:
    """-> {"tables", "doc", "images", "markdown"}. covers: page -> gate output;
    grids matched to table elements by (page, reading_order)."""
    stamp = {
        "package": f"bebop {__version__}",
        "reader": config.READER_MODEL,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "document_id": doc.id,
        "source": doc.source,
    }
    by_element = {}
    for g in grids:
        for owner in (g.element, *g.mates):
            by_element[(owner.page, owner.reading_order)] = g
    emitted: dict[int, str] = {}  # id(grid) -> table id, once shipped
    image_ids: dict[tuple[int, int], str] = {}  # (page, reading_order) -> image id
    parts: list[tuple[dict, LayoutElement]] = []  # caption/footnote blocks to link
    tables: list[dict] = []
    blocks: list[dict] = []
    images: list[dict] = []
    md: list[str] = []
    for pno in sorted(covers):
        page = doc.pages[pno - 1]
        cov = covers[pno]
        for el in sorted(cov.elements, key=lambda e: e.reading_order):
            if el.type == "table":
                grid = by_element.get((el.page, el.reading_order))
                if grid is None:
                    _emit_prose(page, el, cov.claims, blocks, md, origin="table_degraded")
                elif id(grid) in emitted:
                    # Far half of a joined spread: keep its place, point at the joined table.
                    blocks.append(
                        {"page": el.page, "bbox": list(el.bbox), "type": "table",
                         "text": None, "table_id": emitted[id(grid)],
                         "origin": "layout"}
                    )
                else:
                    emitted[id(grid)] = _emit_table(doc, grid, tables, blocks, md)
            elif el.type == "image":
                iid = f"img{len(images) + 1}"
                image_ids[(el.page, el.reading_order)] = iid
                images.append({"id": iid, "page": el.page, "bbox": list(el.bbox)})
                blocks.append(
                    {"page": el.page, "bbox": list(el.bbox), "type": "image",
                     "text": None, "table_id": None, "image_id": iid,
                     "origin": _origin(el)}
                )
                # an image region can own text-layer words (a chart's axis
                # labels and title) — they ship as prose right behind the
                # image block; an image must never silence ink
                _emit_prose(page, el, cov.claims, blocks, md, origin=_origin(el))
            else:  # text | title | list | caption | footnote — the prose lane
                block = _emit_prose(
                    page, el, cov.claims, blocks, md, origin=_origin(el),
                    btype=_PROSE_BLOCK_TYPE.get(el.type, "paragraph"),
                )
                if block is not None and el.type in ("caption", "footnote"):
                    parts.append((block, el))
    # Second pass: a caption usually precedes its table/image in reading
    # order, so the owner's id exists only once the page loop is done.
    for block, el in parts:
        owner = (el.page, el.owner)
        grid = by_element.get(owner)
        # None when the owner shipped no table (a degraded read) — the
        # part is still the layout model's caption, of nothing in tables.json
        block[f"{el.type}_of"] = (
            emitted.get(id(grid)) if grid is not None else image_ids.get(owner)
        )
    return {
        "tables": {"provenance": stamp, "tables": tables},
        "doc": {"provenance": stamp, "pages": len(doc.pages),
                "page_sizes": [_page_size(p) for p in doc.pages], "blocks": blocks},
        # Image inventory entries only; byte extraction is the caller's concern.
        "images": {"provenance": stamp, "images": images},
        "markdown": "\n\n".join(md) + "\n",
    }


def _page_size(page: Page) -> dict:
    """One page's size in PDF points, top-left origin — the space every
    bbox in these assets is in, so an overlay scales by it directly (#701).
    The model's 0.0 (unknown) ships as None, never a stand-in size."""
    return {"number": page.number,
            "width": page.width if page.width > 0 else None,
            "height": page.height if page.height > 0 else None}


def _origin(el: LayoutElement) -> str:
    return "gate" if isinstance(el, GateElement) else "layout"


# Prose element type -> doc.json block type (#700). Every other prose
# element — text, caption, footnote, a gate rescue — is a "paragraph";
# a caption/footnote additionally carries `caption_of` / `footnote_of`.
_PROSE_BLOCK_TYPE = {"title": "heading", "list": "list"}



def _fill(page: Page, element: LayoutElement, claims: dict[int, int]) -> str:
    """The element's claimed text-layer words, verbatim, in (line, x) order."""
    line_of = {gi: ln.id for ln in page.lines for gi in ln.words}
    mine = sorted(
        (i for i, o in claims.items() if o == element.reading_order),
        key=lambda gi: (line_of[gi], page.words[gi].bbox[0], gi),
    )
    lines: list[list[str]] = []
    last: int | None = None
    for gi in mine:
        if line_of[gi] != last:
            lines.append([])
            last = line_of[gi]
        lines[-1].append(page.words[gi].text)
    return "\n".join(" ".join(ws) for ws in lines)

def _emit_prose(
    page, el, claims, blocks, md, origin: str, btype: str = "paragraph"
) -> dict | None:
    """Append the element's prose block (None when it claimed no words)."""
    text = _fill(page, el, claims)
    if not text:
        return None
    block = {"page": el.page, "bbox": list(el.bbox), "type": btype,
             "text": text, "table_id": None, "origin": origin}
    blocks.append(block)
    md.append(text)
    return block


def _emit_table(doc: Document, grid: Grid, tables, blocks, md) -> str:
    entry = _table_entry(doc, grid, f"t{len(tables) + 1}")
    tables.append(entry)
    blocks.append(
        {"page": grid.element.page, "bbox": entry["bbox"], "type": "table",
         "text": None, "table_id": entry["id"], "origin": "layout"}
    )
    table_md = _table_md(entry)
    if table_md:
        md.append(table_md)
    return entry["id"]


def _table_entry(doc: Document, grid: Grid, tid: str) -> dict:
    width = max(
        (len(r.cells) for r in grid.rows if r.kind in ("data", "header")), default=0
    )
    header_kind = [r for r in grid.rows if r.kind == "header"]
    headers = [
        " ".join(
            _text(r.cells[ci])
            for r in header_kind
            if ci < len(r.cells) and _text(r.cells[ci])
        )
        for ci in range(width)
    ]
    # No printed header band is [], never width-many blanks (downstream tests truthiness).
    if not any(headers):
        headers = []
    # Headers live only in `headers`; header_rows stays in the contract, always empty.
    rows, cells = [], []
    for r in grid.rows:
        if r.kind == "header":
            continue
        rows.append([_value(c) for c in r.cells])
        cells.append([_provenance(doc, c, "anchored") for c in r.cells])
    return {
        "id": tid,
        "page": grid.element.page,
        "bbox": list(grid.element.bbox),  # the layout region is the structure authority
        "headers": headers,
        "header_rows": [],
        "rows": rows,
        "cells": cells,
        # In-element ink no cell claimed — evidence with provenance, never
        # rows (descale ruling, 2026-09-01). The invariant this buys:
        # every rows-matrix entry traces to a model transcription.
        "unclaimed": [
            {"text": c.text, "page": c.page, "bbox": _unclaimed_bbox(doc, c)}
            for c in grid.unclaimed
        ],
    }


def _unclaimed_bbox(doc: Document, c) -> list:
    w = doc.pages[c.page - 1].words[c.claims[0]]
    return list(w.bbox)


def _provenance(doc: Document, c: Cell, origin: str) -> dict | None:
    if not c.claims or c.page is None:
        return None
    words = [doc.pages[c.page - 1].words[gi] for gi in c.claims]
    return {
        "origin": origin,
        "page": c.page,
        "bbox": [
            min(w.bbox[0] for w in words),
            min(w.bbox[1] for w in words),
            max(w.bbox[2] for w in words),
            max(w.bbox[3] for w in words),
        ],
    }


def _table_md(entry: dict) -> str:
    # width spans headers AND rows — a header band wider than every data
    # row must widen the separator, or the table breaks as markdown
    width = max((len(r) for r in entry["rows"]), default=0)
    width = max(width, len(entry["headers"]))
    if not width:
        return ""
    # GFM syntax requires the header line even for a table that prints
    # none — blank header cells here are markdown grammar, not a claim
    # about content (tables.json `headers` stays [] for those)
    head = entry["headers"] + [""] * (width - len(entry["headers"]))
    out = [
        "| " + " | ".join(_md_cell(h) for h in head) + " |",
        "|" + "---|" * width,
    ]
    for row in entry["rows"]:
        padded = [c or "" for c in row] + [""] * (width - len(row))
        out.append("| " + " | ".join(_md_cell(c) for c in padded) + " |")
    return "\n".join(out)


def _md_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _text(c: Cell) -> str:
    return c.text.strip()


def _value(c: Cell) -> str | None:
    """Rows-matrix value for one cell. A numeric the page denies (anchoring
    attempted, nothing claimed) ships as None; its text survives in the flag."""
    text = _text(c)
    if text and c.total > 0 and not c.claims and any(ch.isdigit() for ch in text):
        return None
    return text or None
