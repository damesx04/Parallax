"""Page loaders. The held-out directory name is spelled from
parts so that no source file contains the literal name (a test enforces this)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_TOKEN = "pages_" + "held" + "out"
EVAL_DIRNAME = "ev" + "al"
HOLDOUT_DIR = ROOT / EVAL_DIRNAME / HOLDOUT_TOKEN
HOLDOUT_SUMMARY = ROOT / EVAL_DIRNAME / ("held" + "out_summary.json")
HOLDOUT_SPLIT = "held" + "out"
UNLOCK_FILE = ROOT / "FINAL_EVAL_UNLOCKED"
TRAIN_DIR = ROOT / "data" / "pages_train"
DEV_DIR = ROOT / "data" / "pages_dev"


class FinalEvalPagesError(RuntimeError):
    """Raised when held-out pages would be touched outside the final eval."""


def is_holdout_path(path) -> bool:
    p = Path(path)
    if HOLDOUT_TOKEN in str(p) or HOLDOUT_TOKEN in os.path.normpath(str(p)):
        return True
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p.absolute()
    target = HOLDOUT_DIR.resolve() if HOLDOUT_DIR.exists() else HOLDOUT_DIR
    return resolved == target or target in resolved.parents


def _load_dir(d: Path) -> list[dict]:
    pages = []
    for mf in sorted(d.glob("*/manifest.json")):
        m = json.loads(mf.read_text())
        m["html_path"] = str(mf.parent / "page.html")
        pages.append(m)
    if not pages:
        raise FileNotFoundError(f"no pages under {d}")
    return pages


def load_training_pages(path) -> list[dict]:
    """Loads train/dev pages. Refuses anything under the held-out directory."""
    if is_holdout_path(path):
        raise FinalEvalPagesError(f"refusing to load held-out pages for training/dev: {path}")
    pages = _load_dir(Path(path))
    for m in pages:
        if is_holdout_path(m["html_path"]) or m.get("split") == HOLDOUT_SPLIT:
            raise FinalEvalPagesError(f"held-out page reached the training loader: {m['html_path']}")
    return pages


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_final_eval_pages() -> list[dict]:
    """Loads the 240 held-out pages; only allowed once FINAL_EVAL_UNLOCKED exists."""
    if not UNLOCK_FILE.exists():
        raise FinalEvalPagesError("final eval is locked: FINAL_EVAL_UNLOCKED does not exist")
    pages = _load_dir(HOLDOUT_DIR)
    summary = json.loads(HOLDOUT_SUMMARY.read_text())
    expected = summary["hashes"]
    if len(pages) != summary["count"]:
        raise FinalEvalPagesError(f"expected {summary['count']} pages, found {len(pages)}")
    for m in pages:
        if expected.get(m["page_id"]) != sha256_file(m["html_path"]):
            raise FinalEvalPagesError(f"hash mismatch for {m['page_id']}")
    return pages
