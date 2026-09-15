"""Measures rollout and update throughput and writes runs/throughput.json with a
wall-clock projection.

    python -m parallax.train.throughput --out runs/throughput.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from parallax.browser import Browser
from parallax.train.grpo import G, Trainer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/throughput.json")
    ap.add_argument("--pages", type=int, default=2)
    args = ap.parse_args(argv)
    t = Trainer(Path("runs/throughput_tmp"), pages_per_update=1, updates=1, checkpoint_every=1000,
                dev_every=1000, dev_n=4, seed=123)
    res = {"G": G, "updates": []}
    turn_times = []
    with Browser(headless=True) as browser:
        for i in range(args.pages):
            e = t.update(browser)
            res["updates"].append({k: e[k] for k in ("rollout_s", "update_s", "total_s", "n_samples", "mem_gb", "mean_tool_calls", "mean_turns_per_episode")})
            # time a real policy+reference forward/backward on this batch's turns regardless of advantage
            page = t.train_pages[t.order[t.cursor - 1]]
            recs = t.rollout_group(browser, page) if not turn_times else []
            t.model.train()
            for rec in recs[:2]:
                for turn in rec["turns"][:2]:
                    if not turn["completion_ids"]:
                        continue
                    t0 = time.time()
                    loss, lp, kl = t.turn_loss(turn, 1.0)
                    loss.backward()
                    torch.mps.synchronize() if torch.backends.mps.is_available() else None
                    turn_times.append(time.time() - t0)
                    t.opt.zero_grad(set_to_none=True)
        # a single greedy dev episode for eval timing
        t0 = time.time()
        t.dev_eval(browser)
        res["dev_eval_s_per_episode"] = (time.time() - t0) / len(t.dev_pages)
    res["s_per_turn_fwd_bwd_with_ref"] = (sum(turn_times) / len(turn_times)) if turn_times else None
    res["turn_times"] = turn_times
    ups = res["updates"]
    per_episode = sum(u["rollout_s"] for u in ups) / len(ups) / G
    s_turn = res["s_per_turn_fwd_bwd_with_ref"] or 20.0
    observed = sum(u["mean_turns_per_episode"] for u in ups) / len(ups)
    turns_per_episode = max(2.5, observed)   # planning constant (floor 2.5), next to the observed mean
    res["observed_turns_per_episode"] = observed
    res["s_per_episode_rollout"] = per_episode
    res["assumed_turns_per_episode"] = turns_per_episode
    # worst case: every group has nonzero advantage so every turn is trained on
    per_page = G * per_episode + G * turns_per_episode * s_turn
    res["s_per_page_worst_case"] = per_page
    res["projections"] = {}
    for P, U in ((1, 120), (2, 60), (2, 80), (2, 120), (4, 40)):
        hours = U * per_page * P / 3600
        res["projections"][f"P{P}_U{U}"] = {"pages_per_update": P, "updates": U, "episodes": P * U * G, "hours_worst_case": round(hours, 2)}
    res["final_eval_hours_3_policies"] = round(3 * 240 * res["dev_eval_s_per_episode"] / 3600, 2)
    res["peak_mem_gb"] = torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
