"""Enumerate the table crops a parse would read (pipeline.lay_out + reader.render)
with their cache keys. Runs MinerU; set MINERU_SERVER_URL to share one engine."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ..stages import read as reader
from ..pipeline import lay_out


@dataclass(frozen=True)
class Crop:
    pdf: str  # absolute path of the source document
    page: int
    reading_order: int
    image_sha: str  # sha256 of the PNG — the cache key's image half
    cache_path: Path  # reader._cache_path(png): where the read will land
    png: bytes  # the rendered region, byte-identical to a parse's

    @property
    def cached(self) -> bool:
        """The reader's own hit rule: an entry exists AND carries raw
        text (a raw:null transport-failure record is never a hit)."""
        return is_cached(self.cache_path)


def is_cached(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("raw") is not None
    except (OSError, ValueError):
        return False


def enumerate_crops(pdf_paths: Iterable[str | Path]) -> Iterator[Crop]:
    """Every table crop of every document, in parse order. Empty clips
    (render -> None) are skipped exactly as read_table skips them."""
    for pdf in pdf_paths:
        pdf = Path(pdf).resolve()
        layout = lay_out(pdf)
        for el in layout.table_elements:
            png = reader.render(pdf, el)
            if png is None:
                continue
            yield Crop(
                pdf=str(pdf), page=el.page, reading_order=el.reading_order,
                image_sha=reader._sha(png), cache_path=reader._cache_path(png),
                png=png,
            )


def find_pdfs(roots: Iterable[str | Path]) -> list[Path]:
    """Files given directly plus every *.pdf under any directory given,
    sorted, de-duplicated by resolved path."""
    seen: dict[Path, None] = {}
    for root in roots:
        root = Path(root)
        if root.is_dir():
            for p in sorted(root.rglob("*.pdf")):
                seen.setdefault(p.resolve(), None)
        elif root.is_file():
            seen.setdefault(root.resolve(), None)
        else:
            raise FileNotFoundError(root)
    return list(seen)
