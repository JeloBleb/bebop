"""Anchor reader cells to text-layer words, then verify every in-element
word is claimed or explicitly accounted for (flagged, never silently absent)."""

import re

from collections import Counter

from ..model import Cell, Row, Word

DICTCELL = re.compile(r"^\{[^{}]*['\"]text['\"]\s*:\s*['\"](.*?)['\"][^{}]*\}$")

# Unicode dialects seen in reader output vs the text layer; folding is comparison-only.
FOLD = str.maketrans(
    {
        "‑": "-", "‐": "-", "−": "-", "–": "-", "—": "-",
        "’": "'", "‘": "'", "“": '"', "”": '"', " ": " ",
    }
)
STRIP = str.maketrans("", "", "$(),%[]")


def cell_text(cell) -> str:
    """Verbatim cell text; only the dict-cell envelope is unwrapped."""
    if cell is None:
        return ""
    s = str(cell)
    m = DICTCELL.match(s.strip())
    return m.group(1) if m else s


def tokens(cell) -> list[str]:
    """Matching tokens for a cell. Literal "null" (any case) is the reader's
    blank-cell convention and yields none — ONLY the literal word: treating
    "none"/"N/A" as blank erased documents that print them (measured bug)."""
    t = cell_text(cell).strip()
    if t.lower() == "null":
        return []
    return t.split()


def stripped(t: str) -> str:
    return t.translate(FOLD).translate(STRIP)


def anchor_cells(
    rows, lines: list[list[tuple[int, Word]]]
) -> tuple[list[list[Cell]], set[int]]:
    """Anchor reader rows to the element's lines [(word index, Word) sorted by x].
    Returns (anchored rows, claimed indices). Blank cells keep total=0, no claims."""
    used: set[int] = set()
    rows_out: list[list[Cell]] = []
    pending: list[Cell] = []
    cur = 0

    # pass 1 — line-local exact
    for row in rows or []:
        row_max = None
        row_cells = []
        for cell in row:
            toks = tokens(cell)
            if not toks:
                row_cells.append(Cell(text=cell_text(cell), total=0))
                continue
            rec = Cell(text=cell_text(cell), total=len(toks))
            for t in toks:
                found = _find_in_window(t, lines, cur, used)
                if found is None:
                    rec.missed.append(t)
                else:
                    gi, lpos = found
                    used.add(gi)
                    rec.claims.append(gi)
                    row_max = lpos if row_max is None else max(row_max, lpos)
            row_cells.append(rec)
        rows_out.append(row_cells)
        pending.extend(rec for rec in row_cells if rec.missed)
        if row_max is not None:
            cur = max(cur, row_max)

    # pass 2 — order-free over leftovers: exact, then folded/stripped
    by_exact: dict[str, list] = {}
    by_strip: dict[str, list] = {}
    for line in lines:
        for gi, w in line:
            if gi in used:
                continue
            by_exact.setdefault(w.text, []).append(gi)
            by_strip.setdefault(stripped(w.text), []).append(gi)

    for rec in pending:
        still = list(rec.missed)
        rec.missed.clear()
        for t in still:
            gi = None
            if by_exact.get(t):
                gi = by_exact[t].pop(0)
            else:
                st = stripped(t)
                if st and by_strip.get(st):
                    gi = by_strip[st].pop(0)
            if gi is None:
                rec.missed.append(t)
            elif gi not in used:
                used.add(gi)
                rec.claims.append(gi)

    return rows_out, used


def _find_in_window(t: str, lines, cur: int, used: set[int]):
    """Exact match in the current Line, then its immediate neighbors."""
    for lpos in (cur, cur + 1, cur - 1):
        if 0 <= lpos < len(lines):
            for gi, w in lines[lpos]:
                if gi not in used and w.text == t:
                    return gi, lpos
    return None


# Financial marks: currency, percent, parenthesized negatives, thousands commas, dates, decimals.
FINANCIAL_MARK = re.compile(r"[\$%]|\(\d|\d,\d|\d/\d|\d\.\d")


def looks_financial(text: str) -> bool:
    return bool(FINANCIAL_MARK.search(text))


def verify_element(
    lines: list[list[tuple[int, Word]]],
    reader_rows,
    *,
    page: int,
) -> tuple[list[Row], list[Cell], list[dict]]:
    """-> (anchored data rows, unclaimed in-element words as Cells, flags)."""
    flags: list[dict] = []
    words = {gi: w for line in lines for gi, w in line}

    anchored, claimed = anchor_cells(reader_rows, lines)
    for row in anchored:
        for c in row:
            c.page = page

    # document-side closure: every unclaimed in-element word is accounted for
    model_ct = Counter(t for row in anchored for c in row for t in c.text.split())
    layer_ct = Counter(w.text for w in words.values())
    unclaimed: list[Cell] = []
    for gi in sorted(words):
        if gi in claimed:
            continue
        w = words[gi]
        if any(ch.isdigit() for ch in w.text) and model_ct[w.text] >= layer_ct[w.text]:
            continue  # numeric duplicate: only instance attribution is ambiguous
        if not any(ch.isalnum() for ch in w.text):  # decorative
            flags.append(
                dict(page=page, kind="furniture_omitted", severity="furniture",
                     detail=w.text[:30], refs=[gi])
            )
        else:
            unclaimed.append(Cell(text=w.text, claims=[gi], total=1, page=page))
            flags.append(
                dict(page=page, kind="layer_unclaimed", severity="low",
                     detail=w.text[:30], refs=[gi])
            )

    # cell-level findings — two kinds, no thresholds (descale round 2):
    # a cell either fully anchored or it didn't, and the difference between
    # "no claims" and "some missed" is all a reader needs to triage.
    for row in anchored:
        for c in row:
            if c.total == 0 or not c.missed:
                continue
            kind = "cell_unanchored" if not c.claims else "cell_partial"
            flags.append(
                dict(page=page, kind=kind, severity="high",
                     detail=c.text[:40], refs=list(c.claims))
            )

    rows_out = [Row(cells=row, kind="data") for row in anchored]
    return rows_out, unclaimed, flags
