"""The matcher bug catalog (ported from v2/tests/test_anchor.py): every
past bug, a permanent test. Only the imports changed — AnchoredCell is
model.Cell now, and the anchor module ports verbatim."""

from bebop.stages.anchor import anchor_cells, cell_text, tokens
from bebop.model import Word
from bebop.stages.anchor import verify_element


def make_lines(*line_texts, start=0):
    """Each arg is one Line's words; word indices run across lines."""
    lines, gi = [], start
    for words in line_texts:
        line, x = [], 72.0
        for t in words:
            line.append((gi, Word(text=t, bbox=(x, 0, x + 8 * len(t), 11), index=gi)))
            x += 8 * len(t) + 12
            gi += 1
        lines.append(line)
    return lines


def flat(rows):
    return [c for row in rows for c in row]


def claims_of(rows):
    return {c.text: c.claims for row in rows for c in row}


def test_exact_line_local_claims_in_order():
    lines = make_lines(["Revenue", "1,234"], ["Expenses", "5,678"])
    rows, used = anchor_cells([["Revenue", "1,234"], ["Expenses", "5,678"]], lines)
    assert used == {0, 1, 2, 3}
    assert all(not c.missed for c in flat(rows))


def test_duplicate_starvation_solved_by_line_windows():
    # "Total" and "1,234" appear on two distant lines; v1's flat scan let the
    # first row steal the far instance. Line-local windows claim locally.
    lines = make_lines(["Total", "1,234"], ["Rent", "99"], ["Total", "1,234"])
    rows, used = anchor_cells([["Total", "1,234"], ["Rent", "99"], ["Total", "1,234"]], lines)
    assert claims_of(rows)  # all matched
    assert [c.claims for c in flat(rows)] == [[0], [1], [2], [3], [4], [5]]


def test_row_located_after_cursor_stall():
    # A junk row the window can't match must not derail later rows: pass 2
    # anchors later rows' tokens on their own lines.
    lines = make_lines(["Header", "Col1"], ["Alpha", "10"], ["Beta", "20"])
    rows, used = anchor_cells(
        [["Header", "Col1"], ["hallucinated", "junk"], ["Beta", "20"]], lines
    )
    assert claims_of(rows)["Beta"] == [4]
    assert claims_of(rows)["20"] == [5]


def test_only_literal_null_is_blank():
    # Treating "none" as blank erased documents printing "None" (measured).
    assert tokens(None) == []
    assert tokens("null") == [] and tokens("NULL") == []
    assert tokens("None") == ["None"]
    assert tokens("N/A") == ["N/A"]


def test_digit_tokens_never_containment_fuzzy():
    # "1,234" must not claim "11,234" via containment (measured guardrail).
    lines = make_lines(["11,234"])
    rows, used = anchor_cells([["1,234"]], lines)
    assert used == set()
    assert flat(rows)[0].missed == ["1,234"]


def test_unicode_hyphen_dialect_folds():
    # Layer prints U+2011 non-breaking hyphen; model emits ASCII (measured).
    lines = make_lines(["Mark‑to‑Market"])
    rows, used = anchor_cells([["Mark-to-Market"]], lines)
    assert used == {0}


def test_dict_cell_envelope_unwraps():
    # The dict-cell incident: model creativity wraps a value in an object.
    assert cell_text('{"text": "1,234", "confidence": 0.9}') == "1,234"
    assert tokens('{"text": "1,234"}') == ["1,234"]


def test_verbatim_text_preserved_typos_and_minus():
    # Colemen typo and the -$14.93 minus trap must survive verbatim.
    lines = make_lines(["Colemen", "-$14.93"])
    rows, used = anchor_cells([["Colemen", "-$14.93"]], lines)
    cells = flat(rows)
    assert cells[0].text == "Colemen" and cells[1].text == "-$14.93"
    assert used == {0, 1}


def test_null_cells_keep_row_position_with_no_claims():
    lines = make_lines(["A", "1"])
    rows, _ = anchor_cells([["A", None, "1"]], lines)
    assert [c.total for c in rows[0]] == [1, 0, 1]
    assert rows[0][1].claims == []


def test_wrapped_row_spans_neighbor_line():
    # A label wrapping onto the next Line is claimable: window includes cur+1.
    lines = make_lines(["Very", "Long", "1,234"], ["Label"])
    rows, used = anchor_cells([["Very Long Label", "1,234"]], lines)
    assert used == {0, 1, 2, 3}


# --- merged from test_verify.py ---

def _line(y, texts, start):
    return [
        (start + i, Word(text=t, bbox=(50 + 60 * i, y, 100 + 60 * i, y + 10), index=start + i))
        for i, t in enumerate(texts)
    ]


def test_unclaimed_ink_is_evidence_never_rows():
    # three printed lines; the reader dropped the MIDDLE line's value —
    # it lands on the unclaimed side-channel with provenance, and the
    # rows matrix holds ONLY what the model transcribed (descale ruling)
    lines = [
        _line(100, ["Metric", "FY24"], start=0),
        _line(114, ["Revenue", "1,234"], start=2),
        _line(128, ["Expenses", "2,345"], start=4),
    ]
    read = (("Metric", "FY24"), ("Revenue",), ("Expenses", "2,345"))
    rows, unclaimed, flags = verify_element(lines, read, page=1)
    assert [r.kind for r in rows] == ["data", "data", "data"]
    assert [c.text for c in unclaimed] == ["1,234"]
    assert unclaimed[0].claims
    assert any(f["kind"] == "layer_unclaimed" for f in flags)
