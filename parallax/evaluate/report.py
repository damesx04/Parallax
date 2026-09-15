"""Writes RESULTS.md from runs/final/metrics.json and the episode logs
(headline M1, per-policy and per-category tables, validity flags). Kept
separate from the metrics module.

    python -m parallax.evaluate.report [runs/final]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from parallax.evaluate.metrics import POLICIES, compute, DEFAULT_RUN_DIR

NAMES = {"baseline_a": "Baseline A (DOM-only)", "baseline_b": "Baseline B (verify-every-step)", "trained": "Trained policy"}


def fmt(x, pct=False):
    if x is None:
        return "undefined"
    return f"{100 * x:.1f}%" if pct else (f"{x:.1f}" if isinstance(x, float) else str(x))


def main(argv=None):
    run_dir = Path(((argv if argv is not None else sys.argv[1:]) or [DEFAULT_RUN_DIR])[0])
    m = compute(run_dir)
    lines = ["# Parallax results (final eval on the 240 held-out pages)", ""]
    lines.append(f"- **M1** (wrong-target clicks, trained vs Baseline A): {fmt(m['M1'])}%" if m["M1"] is not None else "- **M1**: undefined (Baseline A had no wrong-target clicks)")
    lines.append(f"- Degenerate flag: **{m['degenerate']}**" + (f" ({'; '.join(m['degenerate_reasons'])})" if m.get("degenerate_reasons") else ""))
    lines += ["", "## Per-policy summary", "", "| Policy | Success | Wrong-target episodes (W) | Clicking episodes (K) | No-click episodes | W/K | Verify calls | Invalid outputs |",
              "|---|---|---|---|---|---|---|---|"]
    for p in POLICIES:
        s = m["policies"][p]
        lines.append(f"| {NAMES[p]} | {fmt(s['success_rate'], True)} | {s['W']} | {s['K']} | {s['N_noclick']} | {fmt(s['W_over_K'], True)} | {s['verify_calls']} | {s['invalid_outputs']} |")
    lines += ["", "## Per-category breakdown (success / wrong-target)", "", "| Category | " + " | ".join(NAMES[p] for p in POLICIES) + " |", "|---|" + "---|" * len(POLICIES)]
    cats = sorted({k for p in POLICIES for k in m["policies"][p]["per_category"]})
    for c in cats:
        cells = []
        for p in POLICIES:
            pc = m["policies"][p]["per_category"].get(c, {"n": 0})
            cells.append(f"{pc.get('success', 0)}/{pc['n']} succ, {pc.get('wrong_target', 0)} wrong")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "## Definitions", "", *[f"- {k}: {v}" for k, v in m["definitions"].items()]]
    Path("RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
