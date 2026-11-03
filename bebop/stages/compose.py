"""Reassemble what layout fragmented: band merge (stacked table elements,
before reading), spread join (facing-page halves) and continuation header
inheritance (after reading). Every rule appends an audit flag."""

from __future__ import annotations

from dataclasses import replace

from .. import config
from ..model import Cell, Grid, LayoutElement, Page, Row
from .anchor import looks_financial


# --- band merge (element level) --------------------------------------------


def merge_bands(
    page: Page, elements: list[LayoutElement], audit: list | None = None
) -> list[LayoutElement]:
    """Merge footprint-aligned stacked table elements into one (union bbox,
    top band's reading_order and header_rows). Appends a bands_merged flag."""
    tables = [e for e in elements if e.type == "table"]
    if len(tables) < 2:
        return elements
    tables.sort(key=lambda e: e.reading_order)

    groups: list[list[LayoutElement]] = []
    for el in tables:
        target = None
        # A band with its own header rows is a table start; only headerless bands join.
        if not el.header_rows:
            for g in groups:
                if _footprints_align(_words_x(page, g[-1]), _words_x(page, el)):
                    target = g
                    break
        if target is not None:
            target.append(el)
        else:
            groups.append([el])

    merged: dict[int, LayoutElement] = {}  # reading_order of any member -> merged element
    for g in groups:
        if len(g) == 1:
            continue
        top = g[0]
        boxes = [e.bbox for e in g]
        union = (
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes),
        )
        m = LayoutElement(
            type="table", page=top.page, bbox=union,
            reading_order=top.reading_order, header_rows=top.header_rows,
        )
        for e in g:
            merged[e.reading_order] = m
        if audit is not None:
            audit.append(dict(
                page=top.page, kind="bands_merged", severity="audit",
                detail=f"{len(g)} table bands merged "
                       f"(orders {sorted(e.reading_order for e in g)})",
                refs=[],
            ))

    out: list[LayoutElement] = []
    seen: set[int] = set()
    for el in sorted(elements, key=lambda e: e.reading_order):
        if el.type == "table" and el.reading_order in merged:
            m = merged[el.reading_order]
            if id(m) not in seen:
                seen.add(id(m))
                out.append(m)
        elif el.owner in merged:
            # a caption/footnote of a merged band follows it into the union
            out.append(replace(el, owner=merged[el.owner].reading_order))
        else:
            out.append(el)
    return out


def _words_x(page: Page, el: LayoutElement) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = el.bbox
    return [
        (w.bbox[0], w.bbox[2])
        for w in page.words
        if x0 <= (w.bbox[0] + w.bbox[2]) / 2 <= x1
        and y0 <= (w.bbox[1] + w.bbox[3]) / 2 <= y1
    ]


def _footprints_align(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    """v1/v2-measured: the majority of one party's ink must lie inside the
    other's column footprint — same-layout stacks coincide, different
    layouts interleave into each other's gaps. Symmetric via max."""
    ia, ib = _intervals(a), _intervals(b)
    if not ia or not ib:
        return False
    return max(_covered(ia, ib), _covered(ib, ia)) >= config.FOOTPRINT_ALIGN_MIN


def _intervals(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for x0, x1 in sorted(spans):
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    return [(a, b) for a, b in merged]


def _covered(sub, cover) -> float:
    total = sum(b - a for a, b in sub)
    hit = sum(max(0.0, min(b, d) - max(a, c)) for a, b in sub for c, d in cover)
    return hit / total if total else 0.0


# --- spread join (grid level) ----------------------------------------------


def join_spreads(
    grids: list[Grid], pages: dict[int, Page], audit: list | None = None
) -> list[Grid]:
    """Join a left-page grid with the facing page's grid when data rows y-pair
    within SPREAD_Y_TOL_PT, the right half leads with financial values, and
    both halves sit against the fold. Left keeps identity; right becomes a mate."""
    out = list(grids)
    by_page: dict[int, list[Grid]] = {}
    for g in out:
        by_page.setdefault(g.element.page, []).append(g)

    for p in sorted(by_page):
        for left in list(by_page.get(p, [])):
            if left not in out:
                continue
            lpage = pages.get(p)
            if lpage is None or lpage.width <= 0:
                continue  # unknown sheet: the fold rule can't hold — refuse
            if left.element.bbox[2] < lpage.width / 2:
                continue  # left half doesn't reach the fold
            ly = _row_ys(left, pages)
            lys = [y for r, y in ly if r.kind == "data" and y is not None]
            if not lys:
                continue
            for right in list(by_page.get(p + 1, [])):
                if right not in out or not _is_continuation(right):
                    continue
                rpage = pages.get(p + 1)
                if rpage is None or rpage.width <= 0:
                    continue
                if right.element.bbox[0] > rpage.width / 2:
                    continue  # right half doesn't start at the fold
                ry = _row_ys(right, pages)
                rys = [y for r, y in ry if r.kind == "data" and y is not None]
                if not rys:
                    continue
                paired = sum(
                    1 for y in lys
                    if any(abs(y - y2) <= config.SPREAD_Y_TOL_PT for y2 in rys)
                )
                if paired < config.SPREAD_MATCH_MIN * min(len(lys), len(rys)):
                    continue
                out[out.index(left)] = _join(left, ly, right, ry)
                out.remove(right)
                if audit is not None:
                    audit.append(dict(
                        page=p, kind="spread_joined", severity="audit",
                        detail=f"facing-page spread joined: p{p} order "
                               f"{left.element.reading_order} + p{p + 1} order "
                               f"{right.element.reading_order} "
                               f"({paired}/{min(len(lys), len(rys))} rows y-paired)",
                        refs=[],
                    ))
                break
    return out


def _row_ys(grid: Grid, pages: dict[int, Page]) -> list[tuple[Row, float | None]]:
    """Each row's y0 on its page — the minimum y of its claimed words
    (v2 rows carried line y; v3 derives it from the ledger)."""
    out = []
    for row in grid.rows:
        ys = [
            pages[c.page].words[gi].bbox[1]
            for c in row.cells
            if c.page is not None and c.page in pages
            for gi in c.claims
        ]
        out.append((row, min(ys) if ys else None))
    return out


def _is_continuation(grid: Grid) -> bool:
    """A spread's right half has no label column: the majority of its data
    rows lead with a financial value (v2-measured)."""
    firsts = []
    for row in grid.rows:
        if row.kind != "data":
            continue
        first = next((c.text for c in row.cells if c.text.strip()), None)
        if first is not None:
            firsts.append(looks_financial(first))
    return bool(firsts) and sum(firsts) / len(firsts) >= 0.5


def _join(left: Grid, ly, right: Grid, ry) -> Grid:
    joined_rows: list[Row] = []
    right_ys = list(ry)
    used: set[int] = set()
    for row, y in ly:
        mate = None
        if y is not None:
            for i, (r2, y2) in enumerate(right_ys):
                if i not in used and y2 is not None and abs(y - y2) <= config.SPREAD_Y_TOL_PT:
                    mate = i
                    break
        if mate is None:
            joined_rows.append(row)
        else:
            used.add(mate)
            r2 = right_ys[mate][0]
            joined_rows.append(
                Row(cells=row.cells + r2.cells,
                    kind="data" if "data" in (row.kind, r2.kind) else row.kind)
            )
    # Unpaired right-half rows pad to the left width so their cells stay in right columns.
    lw = max((len(r.cells) for r, _ in ly), default=0)
    joined_rows.extend(
        Row(cells=[Cell(text="") for _ in range(lw)] + r.cells, kind=r.kind)
        for i, (r, _) in enumerate(right_ys)
        if i not in used
    )
    # Both halves' findings and identities survive the join: flags feed the
    # pipeline's collection (which runs after joining), mates tell emit the
    # right element's content lives in this grid.
    return Grid(element=left.element, rows=joined_rows,
                flags=left.flags + right.flags,
                mates=left.mates + (right.element,) + right.mates,
                unclaimed=left.unclaimed + right.unclaimed)


# --- continuation header inheritance (grid level) --------------------------


def inherit_headers(
    grids: list[Grid], pages: dict[int, Page], audit: list | None = None
) -> None:
    """A headerless grid adopts the printed header rows of the nearest earlier
    grid with the same modal width, edges within CONTINUATION_X_TOL_PT and
    aligned footprints. Inherited headers never propagate; provenance stays."""
    ordered = sorted(grids, key=lambda g: (g.element.page, g.element.reading_order))
    sources = [g for g in ordered if _printed_header_rows(g)]  # pre-inheritance snapshot
    for target in ordered:
        if _printed_header_rows(target) or target.element.header_rows:
            continue
        twidth = _modal_width(target)
        if not twidth:
            continue
        candidates = [s for s in sources if s.element.page < target.element.page]
        for src in sorted(
            candidates,
            key=lambda s: (s.element.page, s.element.reading_order),
            reverse=True,
        ):
            if _modal_width(src) != twidth:
                continue
            dx0 = abs(target.element.bbox[0] - src.element.bbox[0])
            dx1 = abs(target.element.bbox[2] - src.element.bbox[2])
            if dx0 > config.CONTINUATION_X_TOL_PT or dx1 > config.CONTINUATION_X_TOL_PT:
                continue
            ia = _claimed_x_intervals(src, pages)
            ib = _claimed_x_intervals(target, pages)
            if not ia or not ib:
                continue
            if max(_covered(ia, ib), _covered(ib, ia)) < config.FOOTPRINT_ALIGN_MIN:
                continue
            target.rows[:0] = [
                Row(cells=list(r.cells), kind="header") for r in src.rows
                if r.kind == "header"
            ]
            if audit is not None:
                audit.append(dict(
                    page=target.element.page, kind="header_inherited",
                    severity="audit",
                    detail=f"headers inherited from p{src.element.page} order "
                           f"{src.element.reading_order} by p{target.element.page} "
                           f"order {target.element.reading_order}",
                    refs=[],
                ))
            break


def _printed_header_rows(grid: Grid) -> bool:
    return any(
        r.kind == "header" and any(c.text.strip() for c in r.cells)
        for r in grid.rows
    )


def _modal_width(grid: Grid) -> int:
    widths = [len(r.cells) for r in grid.rows if r.kind == "data"]
    if not widths:
        return 0
    # deterministic modal: most frequent, widest on a tie
    return max(set(widths), key=lambda w: (widths.count(w), w))


def _claimed_x_intervals(grid: Grid, pages: dict[int, Page]) -> list[tuple[float, float]]:
    return _intervals([
        (pages[c.page].words[gi].bbox[0], pages[c.page].words[gi].bbox[2])
        for row in grid.rows
        for c in row.cells
        if c.page is not None and c.page in pages
        for gi in c.claims
    ])
