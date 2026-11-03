"""Coverage gate: the invariant (every word claimed exactly once), snap,
clustering, and the degrade-never-delete rules."""

from bebop import config
from bebop.stages.cover import cover
from bebop.stages.ingest import ingest
from bebop.model import GateElement


def hull(words):
    return (
        min(w.bbox[0] for w in words),
        min(w.bbox[1] for w in words),
        max(w.bbox[2] for w in words),
        max(w.bbox[3] for w in words),
    )


def test_full_cover_yields_no_gates(make_pdf, element):
    page = ingest(make_pdf([(72, 100, "Net Operating Income")])).pages[0]
    el = element(bbox=(36, 60, 576, 140), order=0)
    cov = cover(page, [el])
    assert not any(isinstance(e, GateElement) for e in cov.elements)
    assert set(cov.claims) == {w.index for w in page.words}
    assert set(cov.claims.values()) == {0}


def test_invariant_every_word_claimed_exactly_once(make_pdf, element):
    # body covered by a proposal; footer words are a layout miss
    page = ingest(
        make_pdf([(72, 100, "body prose here"), (72, 750, "Page 1 of 30")])
    ).pages[0]
    cov = cover(page, [element(bbox=(36, 60, 576, 140), order=0)])
    # the invariant: proposals + gates partition the text layer
    assert set(cov.claims) == {w.index for w in page.words}
    gates = [e for e in cov.elements if isinstance(e, GateElement)]
    assert len(gates) == 1
    rescued = {page.words[i].text for i in gates[0].words}
    assert rescued == {"Page", "1", "of", "30"}
    # gate members and the claims map agree — one home for ownership
    assert all(cov.claims[i] == gates[0].reading_order for i in gates[0].words)


def test_snap_tightens_to_word_hull_plus_margin(make_pdf, element):
    page = ingest(make_pdf([(72, 100, "alpha beta")])).pages[0]
    cov = cover(page, [element(bbox=(0, 0, 612, 400), order=0, header_rows=2)])
    x0, y0, x1, y1 = hull(page.words)
    m = config.SNAP_MARGIN_PT
    assert cov.elements[0].bbox == (x0 - m, y0 - m, x1 + m, y1 + m)
    # snapping rebuilds the element; the layout model's header opinion
    # must survive it (reader._mark_headers consumes element.header_rows)
    assert cov.elements[0].header_rows == 2


def test_image_with_words_keeps_its_proposed_bbox(make_pdf, element):
    # a chart claims its axis labels, but its extent is the GRAPHIC —
    # snapping would crop the images inventory down to the text
    page = ingest(make_pdf([(320, 540, "NOI by year")])).pages[0]
    img = element(type="image", bbox=(300, 500, 500, 650), order=0)
    cov = cover(page, [img])
    assert cov.elements[0].bbox == img.bbox
    assert set(cov.claims.values()) == {0}  # the words are still owned


def test_wordless_element_passes_through_untouched(make_pdf, element):
    # an image proposal over blank canvas has nothing to snap to; it must
    # survive with its proposed bbox (a miss degrades, never deletes)
    page = ingest(make_pdf([(72, 100, "caption words")])).pages[0]
    img = element(type="image", bbox=(300, 500, 500, 650), order=1)
    cov = cover(page, [element(bbox=(36, 60, 576, 140), order=0), img])
    assert cov.elements[1] == img
    assert 1 not in cov.claims.values()


def test_separated_misses_become_separate_gates(make_pdf, element):
    page = ingest(make_pdf([(72, 100, "top stray"), (72, 700, "bottom stray")])).pages[0]
    cov = cover(page, [])
    gates = [e for e in cov.elements if isinstance(e, GateElement)]
    assert len(gates) == 2
    assert [page.words[i].text for i in gates[0].words] == ["top", "stray"]
    assert [page.words[i].text for i in gates[1].words] == ["bottom", "stray"]
    # gates order among themselves by position, after all proposals
    assert [g.reading_order for g in gates] == [0, 1]


def test_nested_element_claims_by_smallest_area(make_pdf, element):
    # a caption element nested inside a table region keeps its own words
    page = ingest(make_pdf([(72, 100, "inner")])).pages[0]
    big = element(type="table", bbox=(36, 36, 576, 400), order=0)
    small = element(type="text", bbox=(60, 80, 150, 115), order=1)
    cov = cover(page, [big, small])
    assert set(cov.claims.values()) == {1}


def test_no_text_layer_page_passes_proposals_through(make_pdf, element):
    page = ingest(make_pdf([])).pages[0]
    assert page.has_text_layer is False
    el = element(type="table", bbox=(36, 36, 576, 400), order=0)
    cov = cover(page, [el])
    assert cov.elements == (el,)  # no snap, no gates — nothing to run against
    assert cov.claims == {}
