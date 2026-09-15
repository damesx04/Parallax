"""Final-eval check. Exit 0 iff every condition holds.

    python -m parallax.evaluate.check [runs/final]
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

from parallax.evaluate.metrics import POLICIES, compute, load_records
from parallax.pages.loader import ROOT, UNLOCK_FILE, load_final_eval_pages

REQUIRED = {"page_id", "category", "variant", "seed", "template", "task", "target_id", "intermediate_id", "steps",
            "tool_calls", "verify_calls", "success", "termination", "wrong_target_clicks", "real_clicks",
            "virtual_time", "clock", "actions", "clicks", "history", "policy", "turns", "target_path",
            "browser_version"}


def main(argv=None):
    run_dir = Path(((argv if argv is not None else sys.argv[1:]) or ["runs/final"])[0])
    problems = []
    # hashes and unlock ordering
    try:
        pages = load_final_eval_pages()
        if len(pages) != 240:
            problems.append(f"expected 240 pages, got {len(pages)}")
    except Exception as exc:
        problems.append(f"held-out pages: {exc}")
    tcheck = ROOT / "runs" / "train" / "training_check.json"
    if not tcheck.exists():
        problems.append("runs/train/training_check.json missing")
    else:
        ts = json.loads(tcheck.read_text()).get("timestamp")
        if not ts:
            problems.append("training_check.json has no timestamp")
        elif not UNLOCK_FILE.exists() or UNLOCK_FILE.stat().st_mtime < datetime.datetime.fromisoformat(ts).timestamp():
            problems.append("FINAL_EVAL_UNLOCKED must post-date the training check's timestamp")
    # logs
    for p in POLICIES:
        recs = load_records(run_dir, p)
        if len(recs) != 240:
            problems.append(f"{p}: {len(recs)} episode logs, expected 240")
        ids = {r.get("page_id") for r in recs}
        if len(ids) != len(recs):
            problems.append(f"{p}: duplicate page ids")
        for r in recs:
            missing = REQUIRED - set(r)
            if missing:
                problems.append(f"{p}/{r.get('page_id')}: missing {sorted(missing)}")
                break
            if r["termination"] not in ("success", "wrong_target", "step_limit"):
                problems.append(f"{p}/{r['page_id']}: bad termination {r['termination']}")
            if r["policy"] != p:
                problems.append(f"{p}/{r['page_id']}: policy field {r['policy']}")
            if p == "baseline_a" and any(a["tool"] not in (None, "click_element") for a in r["actions"]):
                problems.append(f"{p}/{r['page_id']}: non click_element action")
            if p == "trained" and not r.get("adapter"):
                problems.append(f"{p}/{r['page_id']}: no adapter recorded")
            if "Chrome/151.0.7922.34" not in (r.get("browser_version") or ""):
                problems.append(f"{p}/{r['page_id']}: browser build not the pinned 151.0.7922.34")
    # metrics recomputation
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        problems.append("metrics.json missing")
    else:
        saved = json.loads(mpath.read_text())
        fresh = compute(run_dir)
        for k in ("M1", "degenerate"):
            if saved.get(k) != fresh.get(k):
                problems.append(f"metrics.json {k}={saved.get(k)} but recomputed {fresh.get(k)}")
        for p in POLICIES:
            for k in ("success_rate", "W", "C", "K", "N_noclick"):
                if saved["policies"][p].get(k) != fresh["policies"][p].get(k):
                    problems.append(f"metrics.json {p}.{k} mismatch")
    for pr in problems:
        print("FAIL:", pr)
    if not problems:
        print("FINAL EVAL CHECK OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
