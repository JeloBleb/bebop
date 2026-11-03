"""``GET /version`` reports the staged tree (#450).

The staging script writes ``BEBOP_TREE`` — the git tree hash of
the ``bebop`` tree it staged — into the service directory. ``/version``
reads it so each ETL run can record which bebop code parsed its document.
Nothing holds on it; an unstaged checkout reports ``null``.
"""

import sys
from pathlib import Path

import pytest

# `service` is a directory beside `bebop`'s modules, not inside the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service import app as app_module  # noqa: E402

TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def test_version_reports_the_staged_tree(tmp_path, monkeypatch):
    staged = tmp_path / "BEBOP_TREE"
    staged.write_text(TREE + "\n")
    monkeypatch.setattr(app_module, "TREE_FILE", staged)

    body = app_module.version()
    assert body["tree"] == TREE
    assert set(body) == {"service", "reader", "tree"}


@pytest.mark.parametrize("content", [None, "", "  \n"])
def test_an_unstaged_or_empty_tree_is_null(tmp_path, monkeypatch, content):
    staged = tmp_path / "BEBOP_TREE"
    if content is not None:
        staged.write_text(content)
    monkeypatch.setattr(app_module, "TREE_FILE", staged)

    assert app_module.version()["tree"] is None


def test_the_tree_file_sits_at_the_staged_service_root():
    """The staging script writes it beside ``service/`` — the root of the
    staged ``bebop`` copy."""
    assert app_module.TREE_FILE.parent == Path(app_module.__file__).resolve().parents[1]
    assert app_module.TREE_FILE.name == "BEBOP_TREE"
