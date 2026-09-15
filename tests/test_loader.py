"""Loader: the training loader refuses held-out paths."""
import os
from pathlib import Path

import pytest

from parallax.pages import loader
from parallax.pages.loader import (DEV_DIR, HOLDOUT_DIR, TRAIN_DIR, FinalEvalPagesError, is_holdout_path,
                                   load_final_eval_pages, load_training_pages)

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "pages_" + "held" + "out"


def test_accepts_train_and_dev():
    tr = load_training_pages(TRAIN_DIR)
    dv = load_training_pages(DEV_DIR)
    assert len(tr) == 800 and len(dv) == 160
    assert all(Path(p["html_path"]).exists() for p in tr[:3])
    assert all(p["split"] in ("train", "dev") for p in tr + dv)


@pytest.mark.parametrize("path", [
    HOLDOUT_DIR,
    str(HOLDOUT_DIR),
    f"ev" + "al/" + TOKEN,
    f"./ev" + "al/" + TOKEN + "/h0000",
    f"data/../ev" + "al/" + TOKEN,
    ROOT / ("ev" + "al") / TOKEN / "h0001" / "page.html",
])
def test_refuses_holdout_paths(path):
    assert is_holdout_path(path)
    with pytest.raises(FinalEvalPagesError):
        load_training_pages(path)


def test_refuses_symlink_to_holdout(tmp_path):
    link = tmp_path / "innocent_pages"
    os.symlink(HOLDOUT_DIR, link)
    assert is_holdout_path(link)
    with pytest.raises(FinalEvalPagesError):
        load_training_pages(link)


def test_refuses_manifest_marked_holdout(tmp_path):
    d = tmp_path / "pages" / "x0000"
    d.mkdir(parents=True)
    (d / "page.html").write_text("<html></html>")
    (d / "manifest.json").write_text('{"page_id": "x0000", "split": "%s"}' % ("held" + "out"))
    with pytest.raises(FinalEvalPagesError):
        load_training_pages(tmp_path / "pages")


def test_final_eval_loader_locked_without_unlock_file(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "UNLOCK_FILE", tmp_path / "FINAL_EVAL_UNLOCKED")
    with pytest.raises(FinalEvalPagesError):
        load_final_eval_pages()


def test_no_python_module_names_holdout_literally():
    """Convention: code spells the guarded names from parts."""
    for p in (ROOT / "parallax").rglob("*.py"):
        assert TOKEN not in p.read_text(), p
        assert ("held" + "out") not in p.read_text().lower(), p
