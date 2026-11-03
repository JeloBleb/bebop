"""Coverage gate: every text-layer word ends up in exactly one element.
Uncovered words cluster into gate elements; proposal bboxes snap to their
claimed words' hull. The claims map is the single owner of word ownership."""

from dataclasses import dataclass, replace
from typing import Sequence

from .. import config
from ..model import Bbox, GateElement, LayoutElement, Page, Word


@dataclass(frozen=True)
class Cover:
    """The gate's output: the page's final element list in reading order
    (snapped proposals, then gate elements) and the word-ownership map
    {word index -> owning element's reading_order}."""

    elements: tuple[LayoutElement, ...]
    claims: dict[int, int]


def cover(page: Page, proposals: Sequence[LayoutElement]) -> Cover:
    ordered = sorted(proposals, key=lambda e: e.reading_order)
    orders = [e.reading_order for e in ordered]
    assert len(set(orders)) == len(orders), "reading_order must be unique per page"

    if not page.has_text_layer:
        return Cover(elements=tuple(ordered), claims={})

    # claim: word center -> smallest-area containing proposal
    claims: dict[int, int] = {}
    for w in page.words:
        best = min(
            (e for e in ordered if _center_in(w, e.bbox)),
            key=lambda e: (_area(e.bbox), e.reading_order),
            default=None,
        )
        if best is not None:
            claims[w.index] = best.reading_order

    # snap: tighten each proposal to its claimed words' hull (+ margin);
    # a word-less element (an image, a scanned region) keeps its bbox —
    # degrade, never delete
    snapped = []
    for e in ordered:
        mine = [page.words[i] for i, o in claims.items() if o == e.reading_order]
        # Images keep their graphic extent, not their words' hull.
        # dataclasses.replace, never a field-by-field rebuild (drops attributes).
        bbox = _pad(_hull(mine)) if mine and e.type != "image" else e.bbox
        snapped.append(replace(e, bbox=bbox))

    # gate: cluster every unclaimed word into fallback elements
    uncovered = [w for w in page.words if w.index not in claims]
    next_order = max(orders, default=-1) + 1
    gates = []
    clusters = sorted(_clusters(uncovered), key=lambda c: (_hull(c)[1], _hull(c)[0]))
    for members in clusters:
        gate = GateElement(
            type="text",
            page=page.number,
            bbox=_pad(_hull(members)),
            reading_order=next_order,
            words=tuple(sorted(w.index for w in members)),
        )
        gates.append(gate)
        for w in members:
            claims[w.index] = gate.reading_order
        next_order += 1

    return Cover(elements=tuple(snapped + gates), claims=claims)


def _clusters(words: list[Word]) -> list[list[Word]]:
    """v1's gate clustering, ported (tools/merge_regions.py): seed on the
    first leftover word, grow the hull while any word sits within
    GATE_ROW_GAP vertically AND GATE_COL_GAP horizontally of it."""
    left = list(words)  # extraction order — deterministic
    out: list[list[Word]] = []
    while left:
        seed = left.pop(0)
        box = list(seed.bbox)
        members = [seed]
        grew = True
        while grew:
            grew = False
            for w in list(left):
                if (
                    w.bbox[1] < box[3] + config.GATE_ROW_GAP_PT
                    and w.bbox[3] > box[1] - config.GATE_ROW_GAP_PT
                    and w.bbox[0] < box[2] + config.GATE_COL_GAP_PT
                    and w.bbox[2] > box[0] - config.GATE_COL_GAP_PT
                ):
                    box = [
                        min(box[0], w.bbox[0]),
                        min(box[1], w.bbox[1]),
                        max(box[2], w.bbox[2]),
                        max(box[3], w.bbox[3]),
                    ]
                    left.remove(w)
                    members.append(w)
                    grew = True
        out.append(members)
    return out


def _center_in(w: Word, b: Bbox) -> bool:
    cx, cy = (w.bbox[0] + w.bbox[2]) / 2, (w.bbox[1] + w.bbox[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def _area(b: Bbox) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def _hull(words: list[Word]) -> Bbox:
    return (
        min(w.bbox[0] for w in words),
        min(w.bbox[1] for w in words),
        max(w.bbox[2] for w in words),
        max(w.bbox[3] for w in words),
    )


def _pad(b: Bbox) -> Bbox:
    m = config.SNAP_MARGIN_PT
    return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)
