"""Composition rules: band merge (footprint continuity), spread join
(facing-page y-pairing + fold rule), continuation header inheritance.

The cross-page tests encode the #99 survey's measured boundaries as
synthetic fixtures — every gate has a passing case and its fail-safe
(criteria don't hold -> fragments ship exactly as read)."""

from bebop.stages.compose import inherit_headers, join_spreads, merge_bands
from bebop.stages.ingest import ingest
from bebop.model import Cell, Grid, LayoutElement, Row


def _el(page, bbox, order, type_="table", header_rows=None):
    return LayoutElement(type=type_, page=page, bbox=bbox,
                         reading_order=order, header_rows=header_rows)


def _bands_pdf(make_pdf):
    # one printed table split as two bands: same three columns, a prose
    # comment between (v1's rent-roll interleave case)
    rows = [(72, 100, "Item"), (300, 100, "Year 1"), (450, 100, "Year 2"),
            (72, 114, "Revenue"), (300, 114, "1,234"), (450, 114, "5,678")]
    rows += [(72, 200, "an interleaved comment paragraph")]
    rows += [(72, 300, "Expenses"), (300, 300, "2,345"), (450, 300, "6,789")]
    return make_pdf(rows)


def test_stacked_aligned_bands_merge_across_interleaved_prose(make_pdf):
    doc = ingest(_bands_pdf(make_pdf))
    page = doc.pages[0]
    els = [
        _el(1, (70, 90, 520, 130), 0, header_rows=1),
        _el(1, (70, 190, 340, 210), 1, type_="text"),
        _el(1, (70, 290, 520, 310), 2),
    ]
    audit = []
    out = merge_bands(page, els, audit=audit)
    tables = [e for e in out if e.type == "table"]
    assert len(tables) == 1
    (m,) = tables
    assert m.bbox == (70, 90, 520, 310)  # union
    assert m.reading_order == 0 and m.header_rows == 1  # top band leads
    assert [e.type for e in out] == ["table", "text"]  # prose untouched
    # the merge is a proposer that can fold distinct printed tables — it
    # must never be silent
    (flag,) = audit
    assert flag["kind"] == "bands_merged" and flag["severity"] == "audit"
    assert flag["page"] == 1 and "orders [0, 2]" in flag["detail"]


def test_part_of_a_merged_band_follows_it_into_the_union(make_pdf):
    # a footnote MinerU attached to the LOWER band: that band's
    # reading_order disappears in the merge, so the part re-points at the
    # merged element rather than dangling
    doc = ingest(_bands_pdf(make_pdf))
    page = doc.pages[0]
    footnote = LayoutElement(type="footnote", page=1, bbox=(70, 190, 340, 210),
                             reading_order=1, owner=2)
    els = [
        _el(1, (70, 90, 520, 130), 0, header_rows=1),
        footnote,
        _el(1, (70, 290, 520, 310), 2),
    ]
    out = merge_bands(page, els)
    assert [(e.type, e.reading_order, e.owner) for e in out] == [
        ("table", 0, None), ("footnote", 1, 0),
    ]


def test_disjoint_footprints_stay_separate(make_pdf):
    # side-by-side panels: x-footprints disjoint, never merged
    rows = [(72, 100, "Left"), (100, 100, "1,234"),
            (400, 100, "Right"), (430, 100, "5,678")]
    doc = ingest(make_pdf(rows))
    page = doc.pages[0]
    els = [_el(1, (70, 90, 200, 120), 0), _el(1, (390, 90, 520, 120), 1)]
    out = merge_bands(page, els)
    assert len([e for e in out if e.type == "table"]) == 2



def test_headered_band_starts_its_own_table(make_pdf):
    # two stacked financial tables with coinciding footprints, EACH with
    # its own header band (wazee p53): model evidence outranks geometry —
    # they must not fold
    doc = ingest(_bands_pdf(make_pdf))
    page = doc.pages[0]
    els = [
        _el(1, (70, 90, 520, 130), 0, header_rows=1),
        _el(1, (70, 290, 520, 310), 1, header_rows=1),
    ]
    audit = []
    out = merge_bands(page, els, audit=audit)
    assert [e.reading_order for e in out if e.type == "table"] == [0, 1]
    assert audit == []


# --- spread join -----------------------------------------------------------
# Pages from make_pdf are 612pt wide -> the fold rule needs the left half
# to end past x=306 and the right half to begin before it.


def _cell(text, page, claims):
    return Cell(text=text, page=page, claims=list(claims))


def _spread_docs(make_pdf, right_rows=None, right_name="r.pdf"):
    left_rows = [(72, 100, "Revenue"), (72, 114, "Expenses")]
    right_rows = right_rows or [(72, 100, "1,234"), (72, 114, "2,345")]
    doc_l = ingest(make_pdf(left_rows, name="l.pdf"))
    doc_r = ingest(make_pdf(right_rows, name=right_name))
    return doc_l, doc_r, {1: doc_l.pages[0], 2: doc_r.pages[0]}


def test_spread_join_pairs_rows_by_claimed_ink(make_pdf):
    # left page: labels; right page: financial columns at the same y's
    _, _, pages = _spread_docs(make_pdf)
    left = Grid(element=_el(1, (70, 90, 560, 130), 0), rows=[
        Row(cells=[_cell("Revenue", 1, [0])], kind="data"),
        Row(cells=[_cell("Expenses", 1, [1])], kind="data"),
    ])
    right = Grid(element=_el(2, (40, 90, 300, 130), 0), rows=[
        Row(cells=[_cell("1,234", 2, [0])], kind="data"),
        Row(cells=[_cell("2,345", 2, [1])], kind="data"),
    ])
    audit = []
    out = join_spreads([left, right], pages, audit=audit)
    assert len(out) == 1
    (j,) = out
    assert [[c.text for c in r.cells] for r in j.rows] == [
        ["Revenue", "1,234"], ["Expenses", "2,345"],
    ]
    assert j.element.page == 1
    # the join is a proposer across pages — never silent, pages named
    (flag,) = audit
    assert flag["kind"] == "spread_joined" and flag["severity"] == "audit"
    assert "p1" in flag["detail"] and "p2" in flag["detail"]


def test_spread_join_carries_flags_and_mates(make_pdf):
    # findings from BOTH halves survive the join (the pipeline collects
    # flags after joining), and the consumed right element rides along as
    # a mate so emit can route its block to the joined table
    _, _, pages = _spread_docs(make_pdf, right_rows=[(72, 100, "1,234")])
    left = Grid(element=_el(1, (70, 90, 560, 130), 0), rows=[
        Row(cells=[_cell("Revenue", 1, [0])], kind="data"),
    ], flags=[{"page": 1, "kind": "left-finding"}])
    right = Grid(element=_el(2, (40, 90, 300, 130), 0), rows=[
        Row(cells=[_cell("1,234", 2, [0])], kind="data"),
    ], flags=[{"page": 2, "kind": "right-finding"}])
    (j,) = join_spreads([left, right], pages)
    assert j.flags == [
        {"page": 1, "kind": "left-finding"},
        {"page": 2, "kind": "right-finding"},
    ]
    assert j.mates == (right.element,)


def test_unpaired_right_rows_keep_their_columns(make_pdf):
    # a right-half row with no y-mate (its header band, say) appends at
    # the bottom OFFSET to the right columns — its cells must never land
    # under the left half's columns
    _, _, pages = _spread_docs(
        make_pdf,
        right_rows=[(72, 60, "FY27"), (72, 100, "1,234"), (72, 114, "2,345")],
    )
    left = Grid(element=_el(1, (70, 90, 560, 130), 0), rows=[
        Row(cells=[_cell("Revenue", 1, [0])], kind="data"),
        Row(cells=[_cell("Expenses", 1, [1])], kind="data"),
    ])
    right = Grid(element=_el(2, (40, 50, 300, 130), 0), rows=[
        Row(cells=[_cell("FY27", 2, [0])], kind="header"),
        Row(cells=[_cell("1,234", 2, [1])], kind="data"),
        Row(cells=[_cell("2,345", 2, [2])], kind="data"),
    ])
    (j,) = join_spreads([left, right], pages)
    assert [[c.text for c in r.cells] for r in j.rows] == [
        ["Revenue", "1,234"], ["Expenses", "2,345"], ["", "FY27"],
    ]
    assert j.rows[2].kind == "header"


def test_self_contained_next_page_table_never_joins(make_pdf):
    # right side leads with labels, not values -> not a continuation
    _, _, pages = _spread_docs(make_pdf, right_rows=[(72, 100, "Other")])
    left = Grid(element=_el(1, (70, 90, 560, 130), 0), rows=[
        Row(cells=[_cell("Revenue", 1, [0])], kind="data"),
    ])
    right = Grid(element=_el(2, (40, 90, 300, 130), 0), rows=[
        Row(cells=[_cell("Other", 2, [0])], kind="data"),
    ])
    assert len(join_spreads([left, right], pages)) == 2


def test_template_repeat_far_from_fold_never_joins(make_pdf):
    # the #99 survey's one false pair above the v2 gates (olive p37/p38):
    # two note panels at coincident template y's, the "right half" 527.8pt
    # into its page. A spread's right half begins at the fold — a panel in
    # the right half of its page is a template repeat, not a continuation.
    _, _, pages = _spread_docs(
        make_pdf, right_rows=[(400, 100, "1,234"), (400, 114, "2,345")]
    )
    left = Grid(element=_el(1, (70, 90, 560, 130), 0), rows=[
        Row(cells=[_cell("Revenue", 1, [0])], kind="data"),
        Row(cells=[_cell("Expenses", 1, [1])], kind="data"),
    ])
    right = Grid(element=_el(2, (390, 90, 600, 130), 0), rows=[
        Row(cells=[_cell("1,234", 2, [0])], kind="data"),
        Row(cells=[_cell("2,345", 2, [1])], kind="data"),
    ])
    audit = []
    assert len(join_spreads([left, right], pages, audit=audit)) == 2
    assert audit == []


def test_left_half_short_of_fold_never_joins(make_pdf):
    # a left table that ends before the fold prints whole on its own page
    # — whatever y-aligns on the next page is not its far half
    _, _, pages = _spread_docs(make_pdf)
    left = Grid(element=_el(1, (70, 90, 200, 130), 0), rows=[
        Row(cells=[_cell("Revenue", 1, [0])], kind="data"),
        Row(cells=[_cell("Expenses", 1, [1])], kind="data"),
    ])
    right = Grid(element=_el(2, (40, 90, 300, 130), 0), rows=[
        Row(cells=[_cell("1,234", 2, [0])], kind="data"),
        Row(cells=[_cell("2,345", 2, [1])], kind="data"),
    ])
    assert len(join_spreads([left, right], pages)) == 2


def test_low_y_pairing_never_joins(make_pdf):
    # rows at different y's: no printed template continues across the
    # fold (SPREAD_MATCH_MIN of the smaller side must pair at 0.5pt)
    _, _, pages = _spread_docs(
        make_pdf, right_rows=[(72, 105, "1,234"), (72, 121, "2,345")]
    )
    left = Grid(element=_el(1, (70, 90, 560, 130), 0), rows=[
        Row(cells=[_cell("Revenue", 1, [0])], kind="data"),
        Row(cells=[_cell("Expenses", 1, [1])], kind="data"),
    ])
    right = Grid(element=_el(2, (40, 90, 300, 130), 0), rows=[
        Row(cells=[_cell("1,234", 2, [0])], kind="data"),
        Row(cells=[_cell("2,345", 2, [1])], kind="data"),
    ])
    assert len(join_spreads([left, right], pages)) == 2


# --- continuation header inheritance ---------------------------------------


def _cont_docs(make_pdf):
    """Source page (1): header band + data. Target page (3): the same
    two-column layout resumed, no header printed. Page 2 is unrelated —
    continuations survive interruption (cedar rent_roll p70 -> p72)."""
    src = ingest(make_pdf(
        [(72, 60, "Suite"), (300, 60, "Rent"),
         (72, 100, "101"), (300, 100, "1,234"),
         (72, 114, "102"), (300, 114, "2,345")], name="src.pdf"))
    tgt = ingest(make_pdf(
        [(72, 100, "201"), (300, 100, "3,456"),
         (72, 114, "202"), (300, 114, "4,567")], name="tgt.pdf"))
    return src, tgt, {1: src.pages[0], 3: tgt.pages[0]}


def _source_grid(order=0, page=1):
    return Grid(element=_el(page, (70, 50, 340, 130), order), rows=[
        Row(cells=[_cell("Suite", page, [0]), _cell("Rent", page, [1])],
            kind="header"),
        Row(cells=[_cell("101", page, [2]), _cell("1,234", page, [3])],
            kind="data"),
        Row(cells=[_cell("102", page, [4]), _cell("2,345", page, [5])],
            kind="data"),
    ])


def _target_grid(element):
    return Grid(element=element, rows=[
        Row(cells=[_cell("201", 3, [0]), _cell("3,456", 3, [1])], kind="data"),
        Row(cells=[_cell("202", 3, [2]), _cell("4,567", 3, [3])], kind="data"),
    ])


def test_headerless_continuation_inherits_headers(make_pdf):
    _, _, pages = _cont_docs(make_pdf)
    src = _source_grid()
    tgt = _target_grid(_el(3, (70, 90, 340, 130), 0))
    audit = []
    inherit_headers([src, tgt], pages, audit=audit)
    assert [c.text for c in tgt.rows[0].cells] == ["Suite", "Rent"]
    assert tgt.rows[0].kind == "header"
    # the header ink lives where it was printed — source-page provenance
    assert all(c.page == 1 for c in tgt.rows[0].cells)
    assert [r.kind for r in tgt.rows] == ["header", "data", "data"]
    assert len(src.rows) == 3  # source untouched
    (flag,) = audit
    assert flag["kind"] == "header_inherited" and flag["severity"] == "audit"
    assert flag["page"] == 3
    assert "p1" in flag["detail"] and "p3" in flag["detail"]


def test_width_mismatch_never_inherits(make_pdf):
    # #99 survey: width equality separated every distinct pairing on the
    # corpus (e.g. cedar p71's 4-wide table between the 10-wide rent-roll
    # halves)
    _, _, pages = _cont_docs(make_pdf)
    src = _source_grid()
    tgt = Grid(element=_el(3, (70, 90, 340, 130), 0), rows=[
        Row(cells=[_cell("201", 3, [0]), _cell("3,456", 3, [1]),
                   _cell("4,567", 3, [3])], kind="data"),
    ])
    audit = []
    inherit_headers([src, tgt], pages, audit=audit)
    assert all(r.kind == "data" for r in tgt.rows)
    assert audit == []


def test_x_offset_never_inherits(make_pdf):
    # measured gap: true continuations agree within 12.3pt, distinct
    # pairings sit at >= 31.5pt — a 30pt shift is a different layout
    _, _, pages = _cont_docs(make_pdf)
    src = _source_grid()
    tgt = _target_grid(_el(3, (100, 90, 370, 130), 0))
    audit = []
    inherit_headers([src, tgt], pages, audit=audit)
    assert all(r.kind == "data" for r in tgt.rows)
    assert audit == []


def test_own_header_evidence_blocks_inheritance(make_pdf):
    # the wazee rule, cross-page: a band declaring its own thead is a
    # table START, never a continuation
    _, _, pages = _cont_docs(make_pdf)
    src = _source_grid()
    tgt = _target_grid(_el(3, (70, 90, 340, 130), 0, header_rows=1))
    audit = []
    inherit_headers([src, tgt], pages, audit=audit)
    assert all(r.kind == "data" for r in tgt.rows)
    assert audit == []


def test_nearest_compatible_source_wins(make_pdf):
    src_doc, _, pages = _cont_docs(make_pdf)
    pages[2] = src_doc.pages[0]  # same layout printed again on page 2
    near = _source_grid(page=2)
    far = _source_grid(page=1)
    tgt = _target_grid(_el(3, (70, 90, 340, 130), 0))
    audit = []
    inherit_headers([far, near, tgt], pages, audit=audit)
    (flag,) = audit
    assert "from p2" in flag["detail"]


def test_inherited_headers_never_propagate(make_pdf):
    # a grid that INHERITED headers is not a source — every continuation
    # traces to a header band the document actually printed (patterson
    # p16 and p17 both inherit from p15)
    src_doc, tgt_doc, pages = _cont_docs(make_pdf)
    pages[2] = tgt_doc.pages[0]  # mid grid: same layout, headerless
    src = _source_grid(page=1)
    mid = Grid(element=_el(2, (70, 90, 340, 130), 0), rows=[
        Row(cells=[_cell("201", 2, [0]), _cell("3,456", 2, [1])], kind="data"),
        Row(cells=[_cell("202", 2, [2]), _cell("4,567", 2, [3])], kind="data"),
    ])
    tgt = _target_grid(_el(3, (70, 90, 340, 130), 0))
    audit = []
    inherit_headers([src, mid, tgt], pages, audit=audit)
    assert [f["page"] for f in audit] == [2, 3]
    assert all("from p1" in f["detail"] for f in audit)


def test_no_compatible_source_leaves_grid_headerless(make_pdf):
    # fail-safe: nothing upstream printed this layout -> the grid ships
    # exactly as read, headers stay an honest []
    _, _, pages = _cont_docs(make_pdf)
    tgt = _target_grid(_el(3, (70, 90, 340, 130), 0))
    audit = []
    inherit_headers([tgt], pages, audit=audit)
    assert all(r.kind == "data" for r in tgt.rows)
    assert audit == []
