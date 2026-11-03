"""PDF -> Document: text-layer words and page-wide reading-order lines.
Pure and deterministic. Lines only order words; the gate owns them."""

import hashlib
from pathlib import Path

import pymupdf

from .. import config
from ..model import Document, Line, Page, Word


def ingest(path: str | Path) -> Document:
    path = Path(path)
    data = path.read_bytes()
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        pages = tuple(_page(pdf[i], i + 1) for i in range(len(pdf)))
    return Document(id=hashlib.sha256(data).hexdigest(), source=path.name, pages=pages)


def _page(page: pymupdf.Page, number: int) -> Page:
    words = tuple(
        Word(text=w[4], bbox=(w[0], w[1], w[2], w[3]), index=i)
        for i, w in enumerate(_dedupe_double_draw(page.get_text("words")))
    )
    lines = tuple(
        Line(id=i, words=tuple(w.index for w in members))
        for i, members in enumerate(_cluster(list(words)))
    )
    return Page(
        number=number,
        words=words,
        lines=lines,
        has_text_layer=len(words) >= config.TEXT_LAYER_MIN_WORDS,
        width=float(page.rect.width),
        height=float(page.rect.height),
    )


def _dedupe_double_draw(raw: list[tuple]) -> list[tuple]:
    """Collapse the same string painted twice at the same spot (double-draw)."""
    survivors: list[tuple] = []
    by_text: dict[str, list[tuple]] = {}
    eps = config.DOUBLE_DRAW_EPSILON_PT
    for w in raw:
        twins = by_text.setdefault(w[4], [])
        if any(all(abs(t[j] - w[j]) <= eps for j in range(4)) for t in twins):
            continue
        twins.append(w)
        survivors.append(w)
    return survivors


def _cluster(words: list[Word]) -> list[list[Word]]:
    """Group words into visual lines by vertical overlap (fraction of the
    shorter height). Lines ordered by y, members by x."""
    bands: list[tuple[float, float, list[Word]]] = []
    for w in sorted(words, key=lambda w: ((w.bbox[1] + w.bbox[3]) / 2, w.bbox[0], w.index)):
        if bands:
            y0, y1, members = bands[-1]
            overlap = min(y1, w.bbox[3]) - max(y0, w.bbox[1])
            shorter = min(y1 - y0, w.bbox[3] - w.bbox[1])
            if overlap >= config.LINE_OVERLAP_MIN * shorter:
                bands[-1] = (min(y0, w.bbox[1]), max(y1, w.bbox[3]), members + [w])
                continue
        bands.append((w.bbox[1], w.bbox[3], [w]))
    return [sorted(m, key=lambda w: (w.bbox[0], w.index)) for _, _, m in bands]
