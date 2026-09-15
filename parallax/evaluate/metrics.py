"""Metric definitions, computed from episode logs only.

    python -m parallax.evaluate.metrics [runs/final]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

POLICIES = ("baseline_a", "baseline_b", "trained")
GUARD_MARGIN = 0.10   # abstention guard, fixed before any run
DEFAULT_RUN_DIR = "runs/final"


def load_records(run_dir: Path, policy: str) -> list[dict]:
    d = Path(run_dir) / policy
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def wrong_target_episodes(records: list[dict]) -> int:
    """W_P: episodes with a wrong-target click."""
    return sum(1 for r in records if r["wrong_target_clicks"] > 0)


def tool_calls(records: list[dict]) -> int:
    """C_P: executed tool calls, verify_target included."""
    return sum(r["tool_calls"] for r in records)


def relative_reduction(trained: int, baseline: int):
    """100 * (1 - trained/baseline); None (undefined) when baseline == 0."""
    if baseline == 0:
        return None
    return 100.0 * (1.0 - trained / baseline)


def summarize(records: list[dict]) -> dict:
    n = len(records)
    per_cat = defaultdict(lambda: {"n": 0, "success": 0, "wrong_target": 0, "tool_calls": 0, "verify_calls": 0})
    for r in records:
        c = per_cat[f"{r.get('category')}_v{r.get('variant')}"]
        c["n"] += 1
        c["success"] += int(bool(r["success"]))
        c["wrong_target"] += int(r["wrong_target_clicks"] > 0)
        c["tool_calls"] += r["tool_calls"]
        c["verify_calls"] += r["verify_calls"]
    succ = sum(int(bool(r["success"])) for r in records)
    # K: episodes with >= 1 target-directed click (a click not on the allowed intermediate)
    K = sum(1 for r in records if any(c.get("outcome") in ("success", "wrong_target") for c in r.get("clicks", [])))
    W = wrong_target_episodes(records)
    return {
        "episodes": n,
        "success": succ,
        "success_rate": (succ / n) if n else None,
        "W": W,
        "K": K,
        "N_noclick": n - K,
        "noclick_rate": ((n - K) / n) if n else None,
        "W_over_K": (W / K) if K else None,
        "C": tool_calls(records),
        "mean_tool_calls": (tool_calls(records) / n) if n else None,
        "verify_calls": sum(r["verify_calls"] for r in records),
        "invalid_outputs": sum(sum(1 for a in r["actions"] if not a.get("valid", a.get("tool") is not None)) for r in records),
        "terminations": {t: sum(1 for r in records if r["termination"] == t) for t in ("success", "wrong_target", "step_limit")},
        "step_limit_rate": (sum(1 for r in records if r["termination"] == "step_limit") / n) if n else None,
        "per_category": dict(per_cat),
    }


def compute(run_dir: Path) -> dict:
    out = {"policies": {}}
    for p in POLICIES:
        out["policies"][p] = summarize(load_records(run_dir, p))
    W_t, W_a = out["policies"]["trained"]["W"], out["policies"]["baseline_a"]["W"]
    out["M1"] = relative_reduction(W_t, W_a)
    s = {p: out["policies"][p]["success_rate"] for p in POLICIES}
    nc = {p: out["policies"][p]["noclick_rate"] for p in POLICIES}
    reasons = []
    if s["trained"] is not None and any(s[b] is not None and s["trained"] < s[b] for b in ("baseline_a", "baseline_b")):
        reasons.append("trained success rate below a baseline's")
    if nc["trained"] is not None and nc["baseline_a"] is not None and nc["trained"] > nc["baseline_a"] + GUARD_MARGIN:
        reasons.append("trained no-click episode rate exceeds baseline A's by more than the margin")
    sl = {p: out["policies"][p]["step_limit_rate"] for p in POLICIES}
    if sl["trained"] is not None and sl["baseline_a"] is not None and sl["trained"] > sl["baseline_a"] + GUARD_MARGIN:
        reasons.append("trained step-limit rate exceeds baseline A's by more than the margin")
    out["degenerate"] = bool(reasons)
    out["degenerate_reasons"] = reasons
    out["definitions"] = {
        "W_P": "episodes with a wrong-target click",
        "M1": "100*(1 - W_trained/W_baselineA), undefined if W_baselineA=0",
        "K_P": "episodes with at least one target-directed click (outcome success or wrong_target)",
        "N_noclick": "episodes with no target-directed click (abstention, incl. dismiss-only stalling)",
        "W_over_K": "wrong-target episodes per clicking episode",
        "degenerate": f"success below a baseline, or no-click rate or step-limit rate > baseline A's + {GUARD_MARGIN}",
        "undefined": "reported as JSON null",
    }
    return out


def main(argv=None):
    run_dir = Path((argv if argv is not None else sys.argv[1:]) or [DEFAULT_RUN_DIR])[0]
    res = compute(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "policies"}, indent=1))
    for p in POLICIES:
        s = res["policies"][p]
        print(p, "episodes", s["episodes"], "success", s["success_rate"], "W", s["W"], "C", s["C"])


if __name__ == "__main__":
    main()
