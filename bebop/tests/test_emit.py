"""Emission: the four-asset contract from the prose path (implemented)
and the table path (hand-built Grid — the reader seam is open)."""

import hashlib

import pytest

from bebop import config
from bebop.stages import read as reader
from bebop.stages.emit import emit
from bebop.stages.cover import cover
from bebop.stages.ingest import ingest
from bebop.model import Cell, Grid, Row, dumps
from bebop.model import GateElement
from bebop.stages.emit import _fill as fill


def wi(page, text):
    return next(w.index for w in page.words if w.text == text)


def cell(page, *texts):
    """A fully-anchored Cell claiming the words printed as `texts`."""
    gis = [wi(page, t) for t in texts]
    joined = " ".join(texts)
    return Cell(text=joined, claims=gis, total=len(gis), page=page.number)


@pytest.fixture
def scene(make_pdf, element):
    """One page: a prose element, a table element (header + data +
    footnote rows), and an uncovered footer that the gate rescues."""
    path = make_pdf(
        [
            (72, 100, "Deal overview text"),
            (72, 200, "Metric"), (300, 200, "FY24"),
            (72, 214, "Revenue"), (300, 214, "1,234"),
            (72, 228, "* excludes credits"),
            (72, 750, "Confidential"),
        ]
    )
    doc = ingest(path)
    page = doc.pages[0]
    cov = cover(
        page,
        [
            element(type="text", bbox=(36, 80, 576, 130), order=0),
            element(type="table", bbox=(60, 180, 400, 245), order=1),
        ],
    )
    table_el = cov.elements[1]
    grid = Grid(
        element=table_el,
        rows=[
            Row(kind="header", cells=[cell(page, "Metric"), cell(page, "FY24")]),
            Row(kind="data", cells=[cell(page, "Revenue"), cell(page, "1,234")]),
        ],
        unclaimed=(cell(page, "excludes"), cell(page, "credits")),
    )
    return doc, page, cov, grid


def test_blocks_in_reading_order_with_origins(scene):
    doc, page, cov, grid = scene
    out = emit(doc, {1: cov}, [grid])
    blocks = out["doc"]["blocks"]
    assert [b["type"] for b in blocks] == ["paragraph", "table", "paragraph"]
    assert [b["origin"] for b in blocks] == ["layout", "layout", "gate"]
    assert blocks[0]["text"] == "Deal overview text"
    assert blocks[1]["table_id"] == "t1"
    assert blocks[2]["text"] == "Confidential"
    assert out["doc"]["pages"] == 1


def test_table_entry_headers_rows_and_provenance(scene):
    doc, page, cov, grid = scene
    entry = emit(doc, {1: cov}, [grid])["tables"]["tables"][0]
    assert entry["id"] == "t1"
    assert entry["headers"] == ["Metric", "FY24"]
    # headers live ONLY in `headers` — never duplicated as a row (the
    # docling/reducto convention; header_rows stays for shape compat)
    assert entry["header_rows"] == []
    assert entry["rows"] == [
        ["Revenue", "1,234"],
    ]
    # the table's bbox is the layout region (snapped by the gate)
    assert entry["bbox"] == list(grid.element.bbox)
    # per-value provenance: origin + page + the claimed word's own bbox
    prov = entry["cells"][0][1]
    assert prov["origin"] == "anchored"
    assert prov["page"] == 1
    assert prov["bbox"] == list(page.words[wi(page, "1,234")].bbox)
    # unclaimed in-element ink ships as evidence, never rows (descale
    # ruling, 2026-09-01): text + provenance on the side-channel
    assert [u["text"] for u in entry["unclaimed"]] == ["excludes", "credits"]
    assert all(u["page"] == 1 and len(u["bbox"]) == 4 for u in entry["unclaimed"])


def test_markdown_is_a_view_of_the_same_output(scene):
    doc, page, cov, grid = scene
    md = emit(doc, {1: cov}, [grid])["markdown"]
    assert "Deal overview text" in md
    assert "| Metric | FY24 |" in md
    assert "| Revenue | 1,234 |" in md
    assert "Confidential" in md
    # header rows render once, in the header line, not again in the body
    assert md.count("Metric") == 1


def test_unanchored_numeric_ships_as_none_not_as_data(scene):
    doc, page, cov, grid = scene
    # a numeric the page denies: anchoring attempted, nothing claimed
    grid.rows[1].cells[1] = Cell(text="($4,410,564)", total=1, claims=[])
    entry = emit(doc, {1: cov}, [grid])["tables"]["tables"][0]
    # zero invented data: the matrix holds None (the model text lives in
    # the cell_unanchored flag); an unanchored LABEL still ships
    assert entry["rows"][0] == ["Revenue", None]
    assert entry["cells"][0][1] is None


def test_image_element_words_ship_as_prose(make_pdf, element):
    # a chart region: an image element that owns text-layer words — the
    # inventory entry AND the words must both ship (ink is never silenced)
    path = make_pdf([(320, 540, "NOI by year")])
    doc = ingest(path)
    page = doc.pages[0]
    cov = cover(page, [element(type="image", bbox=(300, 500, 500, 650), order=0)])
    out = emit(doc, {1: cov}, [])
    assert len(out["images"]["images"]) == 1
    types = [(b["type"], b["text"]) for b in out["doc"]["blocks"]]
    assert types == [("image", None), ("paragraph", "NOI by year")]
    assert "NOI by year" in out["markdown"]


def test_image_block_carries_its_inventory_id(make_pdf, element):
    # the image block links to the images.json entry it describes (#701)
    doc = ingest(make_pdf([(320, 540, "NOI by year")]))
    cov = cover(doc.pages[0], [element(type="image", bbox=(300, 500, 500, 650), order=0)])
    out = emit(doc, {1: cov}, [])
    (image,) = out["images"]["images"]
    (block,) = [b for b in out["doc"]["blocks"] if b["type"] == "image"]
    assert block["image_id"] == image["id"]


def test_page_sizes_ship_in_points(make_pdf):
    # the same PDF-point space as every bbox, so an overlay scales by it (#701)
    doc = ingest(make_pdf([(72, 100, "Deal overview text")]))
    out = emit(doc, {}, [])
    assert out["doc"]["pages"] == 1
    assert out["doc"]["page_sizes"] == [{"number": 1, "width": 612.0, "height": 792.0}]


def test_unknown_page_size_ships_as_none(make_pdf):
    # the model's 0.0 means unknown — never a stand-in size on the wire
    from dataclasses import replace

    doc = ingest(make_pdf([(72, 100, "Deal overview text")]))
    doc = replace(doc, pages=(replace(doc.pages[0], width=0.0, height=0.0),))
    out = emit(doc, {}, [])
    assert out["doc"]["page_sizes"] == [{"number": 1, "width": None, "height": None}]


def test_headerless_table_emits_empty_headers(scene):
    doc, page, cov, grid = scene
    for row in grid.rows:
        if row.kind == "header":
            row.kind = "data"
    out = emit(doc, {1: cov}, [grid])
    entry = out["tables"]["tables"][0]
    # "no header band printed" is [], never width-many blank strings —
    # blanks read as a detection failure downstream
    assert entry["headers"] == []
    assert entry["rows"][0] == ["Metric", "FY24"]
    # the markdown view keeps its (blank) header line — GFM grammar
    # requires one — and the content renders once, in the body
    md = out["markdown"]
    assert "| Revenue | 1,234 |" in md
    assert md.count("Metric") == 1



def test_spread_mate_block_points_at_joined_table(make_pdf, element):
    # two table elements; the second's grid was consumed into the first
    # by the spread join — its block must reference the joined table, not
    # degrade to duplicate prose
    path = make_pdf(
        [(72, 100, "Revenue"), (300, 100, "1,234"), (72, 400, "9,876")]
    )
    doc = ingest(path)
    page = doc.pages[0]
    cov = cover(
        page,
        [
            element(type="table", bbox=(60, 90, 400, 120), order=0),
            element(type="table", bbox=(60, 390, 400, 420), order=1),
        ],
    )
    left_el, right_el = cov.elements
    grid = Grid(
        element=left_el,
        rows=[
            Row(kind="data",
                cells=[cell(page, "Revenue"), cell(page, "1,234"), cell(page, "9,876")]),
        ],
        mates=(right_el,),
    )
    out = emit(doc, {1: cov}, [grid])
    assert len(out["tables"]["tables"]) == 1
    table_blocks = [b for b in out["doc"]["blocks"] if b["type"] == "table"]
    assert [b["table_id"] for b in table_blocks] == ["t1", "t1"]
    assert table_blocks[1]["page"] == right_el.page
    assert all(b["origin"] != "table_degraded" for b in out["doc"]["blocks"])
    assert out["markdown"].count("9,876") == 1


def test_missing_grid_degrades_to_marked_prose(scene):
    doc, page, cov, grid = scene
    out = emit(doc, {1: cov}, [])  # reader seam open: no grids at all
    assert out["tables"]["tables"] == []
    degraded = out["doc"]["blocks"][1]
    assert degraded["type"] == "paragraph"
    assert degraded["origin"] == "table_degraded"
    # content survives verbatim from the text layer — a miss costs
    # structure, never words
    assert degraded["text"] == "Metric FY24\nRevenue 1,234\n* excludes credits"


def test_provenance_stamp_on_every_asset(scene):
    doc, page, cov, grid = scene
    out = emit(doc, {1: cov}, [grid])
    for asset in (out["tables"], out["doc"]):
        stamp = asset["provenance"]
        assert stamp["document_id"] == doc.id
        assert stamp["source"] == "page.pdf"
        assert stamp["package"].startswith("bebop 3.")
        assert stamp["reader"] == config.READER_MODEL
        assert stamp["prompt_sha256"] == hashlib.sha256(reader.PROMPT.encode()).hexdigest()


def test_emission_is_deterministic(scene):
    doc, page, cov, grid = scene
    assert dumps(emit(doc, {1: cov}, [grid])) == dumps(emit(doc, {1: cov}, [grid]))


# --- merged from test_prose.py ---

def test_lines_and_newlines(make_pdf, element):
    page = ingest(
        make_pdf([(72, 100, "first printed line"), (72, 114, "second printed line")])
    ).pages[0]
    el = element(bbox=(36, 60, 576, 140), order=0)
    cov = cover(page, [el])
    assert fill(page, cov.elements[0], cov.claims) == "first printed line\nsecond printed line"


def test_words_sorted_by_x_within_a_line(make_pdf, element):
    # placements inserted out of order; the text layer's geometry decides
    page = ingest(make_pdf([(400, 100, "1,234"), (72, 100, "Revenue"), (250, 100, "Total")])).pages[0]
    el = element(bbox=(36, 60, 576, 140), order=0)
    cov = cover(page, [el])
    assert fill(page, cov.elements[0], cov.claims) == "Revenue Total 1,234"


def test_only_the_elements_own_words_are_included(make_pdf, element):
    page = ingest(make_pdf([(72, 100, "left panel"), (400, 100, "right panel")])).pages[0]
    left = element(bbox=(36, 60, 300, 140), order=0)
    right = element(bbox=(350, 60, 576, 140), order=1)
    cov = cover(page, [left, right])
    assert fill(page, cov.elements[0], cov.claims) == "left panel"
    assert fill(page, cov.elements[1], cov.claims) == "right panel"


def test_gate_element_fills_through_the_same_path(make_pdf, element):
    page = ingest(make_pdf([(72, 100, "covered text"), (72, 750, "Confidential Memorandum")])).pages[0]
    cov = cover(page, [element(bbox=(36, 60, 576, 140), order=0)])
    gate = next(e for e in cov.elements if isinstance(e, GateElement))
    assert fill(page, gate, cov.claims) == "Confidential Memorandum"


def test_wordless_element_yields_empty_string(make_pdf, element):
    page = ingest(make_pdf([(72, 100, "elsewhere")])).pages[0]
    cov = cover(page, [element(bbox=(36, 60, 576, 140), order=0),
                       element(bbox=(300, 500, 500, 650), order=1)])
    assert fill(page, cov.elements[1], cov.claims) == ""
