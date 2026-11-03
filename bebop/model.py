"""Shared types. Word identity is (page, extraction index); cell text is
stored verbatim. dumps() gives deterministic JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass

# x0, y0, x1, y1 — PDF points, top-left origin.
Bbox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Word:
    text: str
    bbox: Bbox
    index: int  # extraction order on the page — the identity, with the page number


@dataclass(frozen=True)
class Line:
    """Page-wide visual line — orders words within an element (never an
    ownership unit, so a line fused across a panel gutter is harmless)."""

    id: int  # unique within the page, ascending y
    words: tuple[int, ...]  # word indices, sorted by x


@dataclass(frozen=True)
class Page:
    number: int  # 1-based
    words: tuple[Word, ...]
    lines: tuple[Line, ...]
    has_text_layer: bool
    # PDF points; 0.0 = unknown, and geometric rules must then refuse.
    width: float = 0.0
    height: float = 0.0


@dataclass(frozen=True)
class Document:
    id: str  # sha256 of the file bytes
    source: str  # file name only — artifacts stay machine-portable
    pages: tuple[Page, ...]


@dataclass(frozen=True)
class LayoutElement:
    """One logical element proposed by the layout model."""

    # table | image | text | title | list | caption | footnote — every
    # role is the layout model's own call; all but table/image are prose
    type: str
    page: int  # 1-based
    bbox: Bbox  # PDF points, top-left origin
    reading_order: int  # dense 0-based per page, the element's identity on it
    # Table elements: the layout model's header row count, None if it offered none.
    header_rows: int | None = None
    # caption/footnote elements: reading_order of the table/image element
    # the layout model attached the part to (same page); None otherwise.
    owner: int | None = None


@dataclass(frozen=True)
class GateElement(LayoutElement):
    """Gate fallback element for words no proposal covered (type "text").
    Carries its member word indices; downstream must not re-derive them."""

    # rescued word indices, sorted (defaulted only by dataclass rule)
    words: tuple[int, ...] = ()


@dataclass
class Cell:
    """The table lane's per-cell record (v2's AnchoredCell, condensed).
    Mutable: anchoring fills claims/missed in place."""

    text: str  # verbatim reader output
    claims: list[int] = field(default_factory=list)  # word indices
    total: int = 0  # tokens attempted
    missed: list[str] = field(default_factory=list)  # tokens that never matched
    page: int | None = None  # the page the claims index into


@dataclass
class Row:
    cells: list[Cell]
    kind: str = "data"  # "header" | "data" — set by the table lane's
    # deterministic line classification; emit never guesses


@dataclass
class Grid:
    """The table lane's output for one table element (LANE 3)."""

    element: LayoutElement
    rows: list[Row]
    # verification findings for this grid (plain dicts from verify_element)
    # — surfaced through the pipeline so no misassociation is ever silent.
    flags: list = field(default_factory=list)
    # Elements consumed into this grid by the spread join; emit points them here.
    mates: tuple[LayoutElement, ...] = ()
    # In-element words no cell claimed: shipped as evidence, never as rows.
    unclaimed: tuple[Cell, ...] = ()


def to_jsonable(obj):
    if is_dataclass(obj):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def dumps(obj) -> str:
    """Deterministic JSON: sorted keys, minimal separators, no ASCII escaping."""
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
