"""MinerU block roles reach doc.json (#700): a title ships as a "heading"
block, a list item as "list", and a caption/footnote MinerU attached to a
table or image as a paragraph carrying `caption_of` / `footnote_of`.
Roles annotate prose; they never filter it — every word still ships."""

from bebop import pipeline
from bebop.stages import propose as mineru
from bebop.stages.cover import cover
from bebop.stages.emit import emit
from bebop.stages.ingest import ingest
from bebop.model import Cell, Grid, LayoutElement, Row


def _cell(page, *texts):
    gis = [next(w.index for w in page.words if w.text == t) for t in texts]
    return Cell(text=" ".join(texts), claims=gis, total=len(gis), page=page.number)


def test_heading_round_trips_from_middle_json_to_doc_json(make_pdf, monkeypatch):
    # The live path minus MinerU itself: a canned middle.json stands in for
    # the subprocess, and parse() runs ingest -> propose -> cover -> emit.
    path = make_pdf([
        (72, 80, "Investment Highlights", 18),
        (72, 120, "Strong tenant credit"),
        (90, 160, "Long-term leases"),
    ])
    middle = {"pdf_info": [{"page_idx": 0, "para_blocks": [
        {"type": "title", "index": 0, "bbox": [60, 55, 450, 90]},
        {"type": "text", "index": 1, "bbox": [60, 105, 500, 130]},
        {"type": "list", "index": 2, "bbox": [60, 145, 500, 170], "blocks": [
            {"type": "text", "index": 2, "bbox": [80, 145, 500, 170]},
        ]},
    ]}]}
    monkeypatch.setattr(mineru, "_middle_json", lambda _path: middle)

    result = pipeline.parse(path)

    blocks = result.assets["doc"]["blocks"]
    assert [(b["type"], b["text"]) for b in blocks] == [
        ("heading", "Investment Highlights"),
        ("paragraph", "Strong tenant credit"),
        ("list", "Long-term leases"),
    ]
    assert all(b["origin"] == "layout" for b in blocks)
    assert not any("caption_of" in b or "footnote_of" in b for b in blocks)


def _caption_scene(make_pdf):
    """A caption printed above a table, the caption element first in
    reading order (MinerU's usual order), attached to the table."""
    path = make_pdf([
        (72, 170, "Rent roll summary"),
        (72, 214, "Revenue"), (300, 214, "1,234"),
        (72, 240, "Source: owner"),
    ])
    doc = ingest(path)
    page = doc.pages[0]
    cov = cover(page, [
        LayoutElement(type="caption", page=1, bbox=(60, 155, 400, 178),
                      reading_order=0, owner=1),
        LayoutElement(type="table", page=1, bbox=(60, 200, 400, 222),
                      reading_order=1),
        LayoutElement(type="footnote", page=1, bbox=(60, 228, 400, 246),
                      reading_order=2, owner=1),
    ])
    grid = Grid(
        element=cov.elements[1],
        rows=[Row(kind="data", cells=[_cell(page, "Revenue"), _cell(page, "1,234")])],
    )
    return doc, cov, grid


def test_caption_and_footnote_link_to_their_table(make_pdf):
    doc, cov, grid = _caption_scene(make_pdf)
    blocks = emit(doc, {1: cov}, [grid])["doc"]["blocks"]
    assert [b["type"] for b in blocks] == ["paragraph", "table", "paragraph"]
    # the caption precedes its table, yet still names the table's id
    assert blocks[0]["text"] == "Rent roll summary"
    assert blocks[0]["caption_of"] == "t1"
    assert "footnote_of" not in blocks[0]
    assert blocks[2]["text"] == "Source: owner"
    assert blocks[2]["footnote_of"] == "t1"
    assert "caption_of" not in blocks[1]


def test_caption_of_a_degraded_table_points_at_nothing(make_pdf):
    # no grid: the table ships as table_degraded prose and tables.json is
    # empty — the caption keeps its role but must not name a table id
    doc, cov, _grid = _caption_scene(make_pdf)
    out = emit(doc, {1: cov}, [])
    assert out["tables"]["tables"] == []
    caption = out["doc"]["blocks"][0]
    assert caption["text"] == "Rent roll summary"
    assert "caption_of" in caption and caption["caption_of"] is None


def test_image_caption_links_to_the_image_id(make_pdf):
    doc = ingest(make_pdf([(72, 520, "Figure 1: Site plan")]))
    page = doc.pages[0]
    cov = cover(page, [
        LayoutElement(type="image", page=1, bbox=(60, 300, 500, 500),
                      reading_order=0),
        LayoutElement(type="caption", page=1, bbox=(60, 505, 500, 528),
                      reading_order=1, owner=0),
    ])
    out = emit(doc, {1: cov}, [])
    (image,) = out["images"]["images"]
    caption = next(b for b in out["doc"]["blocks"] if b["text"])
    assert caption["type"] == "paragraph"
    assert caption["caption_of"] == image["id"]
