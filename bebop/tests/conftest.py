import sys
from pathlib import Path

import pymupdf
import pytest

# Zero-install: make `import bebop` resolve to this service tree's package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def make_pdf(tmp_path):
    """Write a one-page PDF with text placed at exact coordinates.

    Takes [(x, y, text)] or [(x, y, text, fontsize)]; y is the baseline.
    Returns the file path. (Same pattern as v2/tests/conftest.py.)
    """

    def _make(placements, name="page.pdf"):
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        for x, y, text, *rest in placements:
            page.insert_text((x, y), text, fontsize=rest[0] if rest else 11)
        path = tmp_path / name
        doc.save(path)
        doc.close()
        return path

    return _make


@pytest.fixture
def element():
    """Shorthand LayoutElement builder."""
    from bebop.model import LayoutElement

    def _make(type="text", page=1, bbox=(0, 0, 612, 792), order=0, header_rows=None):
        return LayoutElement(type=type, page=page, bbox=tuple(map(float, bbox)),
                             reading_order=order, header_rows=header_rows)

    return _make
