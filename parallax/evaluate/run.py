"""Runs one policy over a page set with greedy decoding, saving one JSON per episode.

    python -m parallax.evaluate.run --policy trained --adapter runs/train/best --split final --out runs/final
    python -m parallax.evaluate.run --policy baseline_a --split dev --limit 20 --out runs/dev_check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from parallax.browser import Browser
from parallax.model.generate import Generator
from parallax.model.load import attach_lora, load_base_model, load_processor
from parallax.model.policies import POLICIES, run_episode
from parallax.pages.loader import DEV_DIR, TRAIN_DIR, load_final_eval_pages, load_training_pages
from parallax.reward import components, reward
from parallax.train.grpo import strip_turns


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=list(POLICIES))
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (trained policy only)")
    ap.add_argument("--split", default="final", choices=["final", "dev", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="runs/final")
    args = ap.parse_args(argv)
    if args.policy == "trained" and not args.adapter:
        print("trained policy requires --adapter", file=sys.stderr)
        return 2
    if args.policy != "trained" and args.adapter:
        print("baselines use the base model without adapters", file=sys.stderr)
        return 2
    if args.split == "final" and args.limit:
        print("--limit is not allowed for the final split (the final eval needs all 240 pages)", file=sys.stderr)
        return 2
    pages = load_final_eval_pages() if args.split == "final" else load_training_pages(DEV_DIR if args.split == "dev" else TRAIN_DIR)
    if args.limit:
        pages = pages[:args.limit]
    processor = load_processor()
    model = load_base_model()
    if args.adapter:
        model = attach_lora(model, args.adapter, trainable=False)
        model.eval()
    gen = Generator(model, processor)
    policy = POLICIES[args.policy](gen, sample=False)
    out = Path(args.out) / args.policy
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Browser(headless=True) as browser:
        for i, page in enumerate(pages):
            dest = out / f"{page['page_id']}.json"
            if dest.exists():
                continue
            rec = run_episode(policy, browser, page)
            rec["reward"] = reward(rec)
            rec["reward_components"] = components(rec)
            rec["adapter"] = args.adapter
            dest.write_text(json.dumps(strip_turns(rec), indent=0))
            if (i + 1) % 10 == 0:
                print(f"{args.policy}: {i + 1}/{len(pages)} ({(time.time() - t0) / (i + 1):.1f} s/episode)", file=sys.stderr, flush=True)
    print(f"done: {len(pages)} episodes in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
