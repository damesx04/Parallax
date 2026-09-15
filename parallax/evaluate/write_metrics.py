"""Writes runs/<dir>/metrics.json using `metrics.compute`.

    python -m parallax.evaluate.write_metrics [runs/final]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from parallax.evaluate.metrics import DEFAULT_RUN_DIR, POLICIES, compute


def main(argv=None):
    run_dir = Path(((argv if argv is not None else sys.argv[1:]) or [DEFAULT_RUN_DIR])[0])
    res = compute(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "policies"}, indent=1))
    for p in POLICIES:
        s = res["policies"][p]
        print(p, "episodes", s["episodes"], "success", s["success_rate"], "W", s["W"], "C", s["C"])


if __name__ == "__main__":
    main()
