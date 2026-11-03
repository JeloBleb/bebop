"""The MinerU adapter's mapping, offline: canned middle.json in, frozen
LayoutElements out. The subprocess/cache path is exercised by the live
pipeline, not unit tests."""

from bebop.stages import propose as mineru


def _flatten_page(para_blocks):
    return mineru._page_elements(para_blocks, 1)


def test_types_map_and_unknowns_degrade_to_text():
    els = _flatten_page([
        {"type": "title", "index": 0, "bbox": [0, 0, 10, 10]},
        {"type": "table", "index": 1, "bbox": [0, 20, 10, 30], "blocks": []},
        {"type": "interline_equation", "index": 2, "bbox": [0, 40, 10, 50]},
    ])
    # MinerU's title is kept as its own role (#700); unknowns still degrade
    assert [e.type for e in els] == ["title", "table", "text"]
    assert all(e.owner is None for e in els)


def test_chart_is_an_image_with_caption_split_out():
    # MinerU 3.4.5 shape: composite chart block nesting chart_body (+
    # optional caption). The graphic joins the images inventory; its words
    # reach prose via the element's claims (emit), never the text lane.
    els = _flatten_page([
        {"type": "chart", "index": 1, "bbox": [0, 10, 99, 99], "blocks": [
            {"type": "chart_body", "index": 1, "bbox": [0, 20, 99, 99]},
            {"type": "chart_caption", "index": 0, "bbox": [0, 10, 99, 18]},
        ]},
    ])
    assert [(e.type, e.reading_order) for e in els] == [("caption", 0), ("image", 1)]
    assert els[0].owner == 1  # attached to the chart's dense reading_order
    assert els[1].bbox == (0, 10, 99, 99)  # the composite bbox


def test_sparse_indices_become_dense_reading_order():
    els = _flatten_page([
        {"type": "text", "index": 7, "bbox": [0, 40, 10, 50]},
        {"type": "text", "index": 2, "bbox": [0, 0, 10, 10]},
        {"type": "text", "bbox": [0, 80, 10, 90]},  # no index -> last
    ])
    assert [e.reading_order for e in els] == [0, 1, 2]
    assert els[0].bbox == (0, 0, 10, 10)      # index 2 first
    assert els[2].bbox == (0, 80, 10, 90)     # index-less last


def test_list_children_become_list_elements():
    els = _flatten_page([
        {"type": "list", "index": 3, "bbox": [0, 0, 99, 99], "blocks": [
            {"type": "text", "index": 3, "bbox": [0, 0, 99, 10]},
            {"type": "text", "index": 4, "bbox": [0, 12, 99, 22]},
        ]},
    ])
    assert [e.type for e in els] == ["list", "list"]
    assert els[0].bbox == (0, 0, 99, 10)


def test_table_header_rows_come_from_html_structure():
    def table(html):
        return {"type": "table", "index": 1, "bbox": [0, 0, 9, 9], "blocks": [
            {"type": "table_body", "index": 1, "bbox": [0, 0, 9, 9], "lines": [
                {"spans": [{"html": html}]}
            ]},
        ]}

    (el,) = _flatten_page([table(
        "<table><thead><tr><th>A</th></tr><tr><th>B</th></tr></thead>"
        "<tr><td>1</td></tr></table>")])
    assert el.header_rows == 2

    (el,) = _flatten_page([table(
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>")])
    assert el.header_rows == 1

    (el,) = _flatten_page([table("<table><tr><td>1</td></tr></table>")])
    assert el.header_rows is None


def test_table_captions_split_out_as_prose_parts():
    els = _flatten_page([
        {"type": "table", "index": 1, "bbox": [0, 10, 99, 99], "blocks": [
            {"type": "table_body", "index": 1, "bbox": [0, 20, 99, 99]},
            {"type": "table_caption", "index": 0, "bbox": [0, 10, 99, 18]},
        ]},
    ])
    assert [(e.type, e.reading_order) for e in els] == [("caption", 0), ("table", 1)]
    assert els[0].owner == 1
    assert els[1].bbox == (0, 10, 99, 99)  # the table keeps the composite bbox


def test_parts_point_at_their_own_owner_across_blocks():
    # Two composites on one page, sparse indices interleaved with prose:
    # each part's owner is ITS composite's dense reading_order, not a
    # position inside the block it came from.
    els = _flatten_page([
        {"type": "title", "index": 0, "bbox": [0, 0, 99, 8]},
        {"type": "table", "index": 3, "bbox": [0, 10, 99, 50], "blocks": [
            {"type": "table_caption", "index": 2, "bbox": [0, 10, 99, 18]},
            {"type": "table_body", "index": 3, "bbox": [0, 20, 99, 45]},
            {"type": "table_footnote", "index": 4, "bbox": [0, 46, 99, 50]},
        ]},
        {"type": "image", "index": 6, "bbox": [0, 60, 99, 99], "blocks": [
            {"type": "image_body", "index": 6, "bbox": [0, 60, 99, 90]},
            {"type": "image_caption", "index": 7, "bbox": [0, 91, 99, 99]},
        ]},
    ])
    assert [(e.type, e.reading_order, e.owner) for e in els] == [
        ("title", 0, None),
        ("caption", 1, 2),
        ("table", 2, None),
        ("footnote", 3, 2),
        ("image", 4, None),
        ("caption", 5, 4),
    ]
