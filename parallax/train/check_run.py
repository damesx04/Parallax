"""Training-run check. Writes runs/train/training_check.json; exit 0 iff it passes.

    python -m parallax.train.check_run runs/train [--planned-updates N]
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="runs/train")
    ap.add_argument("--planned-updates", type=int, default=None)
    args = ap.parse_args(argv)
    d = Path(args.run_dir)
    problems, info = [], {}
    log = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines()] if (d / "log.jsonl").exists() else []
    dev = [json.loads(l) for l in (d / "dev.jsonl").read_text().splitlines()] if (d / "dev.jsonl").exists() else []
    if not log:
        problems.append("no training log")
    trained = [e for e in log if e.get("n_samples", 0) > 0]
    if any(e["loss"] is None or not math.isfinite(e["loss"]) or not math.isfinite(e["kl"]) for e in trained):
        problems.append("non-finite loss/kl in log")
    if not trained:
        problems.append("no update ever trained on a sample (all groups had equal rewards)")
    info["updates"] = len(log)
    info["updates_with_samples"] = len(trained)
    planned = args.planned_updates
    notes = json.loads((d / "plan.json").read_text()) if (d / "plan.json").exists() else {}
    planned = planned or notes.get("planned_updates")
    info["planned_updates"] = planned
    if planned and len(log) < planned and not notes.get("early_stop_reason"):
        problems.append(f"{len(log)} updates < planned {planned} and no documented early stop")
    best = d / "best"
    if not (best / "adapter_model.safetensors").exists():
        problems.append("best/ adapter missing")
    else:
        try:
            from parallax.model.load import attach_lora, load_base_model
            m = attach_lora(load_base_model(), str(best), trainable=False)
            info["best_adapter_loads"] = True
            del m
        except Exception as exc:
            problems.append(f"best adapter failed to load: {exc}")
    # non-triviality: the best adapter must have moved (LoRA B starts at 0) and must behave differently
    # from the base model on the dev subset (an identity adapter would otherwise pass this check)
    if (best / "adapter_model.safetensors").exists():
        from safetensors.torch import load_file
        sd = load_file(best / "adapter_model.safetensors")
        bmax = max(float(v.abs().max()) for k, v in sd.items() if "lora_B" in k)
        info["best_lora_B_max"] = bmax
        if bmax < 1e-3:
            problems.append(f"best adapter is (near) identity: max |lora_B| = {bmax:.2e}")
        state0 = d / "dev_episodes" / "step0000.json"
        st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
        bstep0 = (st.get("best") or {}).get("step")
        stateb = d / "dev_episodes" / f"step{bstep0:04d}.json" if bstep0 is not None else None
        if bstep0 == 0:
            problems.append("best checkpoint is step 0 (the base model): nothing was learned")
        elif stateb and stateb.exists() and state0.exists():
            same = json.dumps(json.load(open(state0)), sort_keys=True) == json.dumps(json.load(open(stateb)), sort_keys=True)
            info["best_dev_episodes_differ_from_base"] = not same
            if same:
                problems.append("best checkpoint's dev episodes are byte-identical to the base model's")
    base = [e for e in dev if e["step"] == 0]
    if not base or not dev:
        problems.append("dev evals missing (need step 0 = base model)")
    else:
        state = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
        bstep = (state.get("best") or {}).get("step")
        best_eval = [e for e in dev if e["step"] == bstep]
        info["base_dev_reward"] = base[0]["mean_reward"]
        info["best_step"] = bstep
        info["best_dev_reward"] = best_eval[-1]["mean_reward"] if best_eval else None
        if best_eval and best_eval[-1]["mean_reward"] < base[0]["mean_reward"]:
            problems.append("best checkpoint's dev reward below the base model's")
    result = {"ok": not problems, "problems": problems, "info": info,
              "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    (d / "training_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
