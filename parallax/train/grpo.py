"""Custom GRPO loop: LoRA policy, group-relative advantages over
G sampled episodes per page, per-turn REINFORCE-style loss with KL to the
frozen base model, one optimizer update per batch.

    python -m parallax.train.grpo --out runs/train --updates 60 --pages-per-update 2
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch

from parallax.browser import Browser
from parallax.model.generate import Generator
from parallax.model.load import attach_lora, load_base_model, load_processor
from parallax.model.policies import TrainedPolicy, run_episode
from parallax.pages.loader import DEV_DIR, TRAIN_DIR, load_training_pages
from parallax.reward import components, reward
from parallax.train.logprobs import completion_logprobs, encode_turn

G = 4
TEMPERATURE = 1.0
LR = 1e-4
KL_BETA = 0.02
GRAD_CLIP = 1.0
ADV_EPS = 1e-4


def prepare_for_training(model) -> list:
    """Gradient checkpointing + input grads for the PEFT model; returns the trainable (LoRA-only) parameters."""
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return [p for p in model.parameters() if p.requires_grad]


def group_advantages(rewards: list[float]) -> list[float]:
    """A_i = (r_i - mean) / (std + ADV_EPS); all zero when every reward is equal."""
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    if std == 0:
        return [0.0] * len(rewards)
    return [(r - mean) / (std + ADV_EPS) for r in rewards]


def strip_turns(rec: dict) -> dict:
    """Episode record without PIL images (for JSON logs)."""
    out = dict(rec)
    out["turns"] = [{k: v for k, v in t.items() if k != "messages"} for t in rec.get("turns", [])]
    return out


def dev_subset(pages: list[dict], n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by_cat = {}
    for p in pages:
        by_cat.setdefault((p["category"], p["variant"]), []).append(p)
    out = []
    i = 0
    while len(out) < n:
        for k in sorted(by_cat):
            if len(out) < n and i < len(by_cat[k]):
                out.append(by_cat[k][i])
        i += 1
        if i > max(len(v) for v in by_cat.values()):
            break
    rng.shuffle(out)
    return out


class Trainer:
    def __init__(self, out_dir: Path, pages_per_update: int, updates: int, checkpoint_every: int,
                 dev_every: int, dev_n: int, seed: int, resume_from: str | None = None):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.P, self.updates = pages_per_update, updates
        self.checkpoint_every, self.dev_every, self.dev_n = checkpoint_every, dev_every, dev_n
        self.seed = seed
        self.processor = load_processor()
        base = load_base_model()
        self.model = attach_lora(base, resume_from, trainable=True)
        self.params = prepare_for_training(self.model)
        self.gen = Generator(self.model, self.processor)
        self.opt = torch.optim.AdamW(self.params, lr=LR, weight_decay=0.0)
        self.train_pages = load_training_pages(TRAIN_DIR)
        self.dev_pages = dev_subset(load_training_pages(DEV_DIR), dev_n)
        self.rng = random.Random(seed)
        self.order = list(range(len(self.train_pages)))
        self.rng.shuffle(self.order)
        self.step = 0
        self.cursor = 0
        self.best = None
        state = self.out / "state.json"
        if resume_from and not state.exists():
            raise RuntimeError("--resume requires the run's state.json in --out (the step-0 dev eval must stay the base model's)")
        if resume_from and state.exists():
            st = json.loads(state.read_text())
            self.step, self.cursor, self.best, self.order = st["step"], st["cursor"], st["best"], st["order"]
            if (Path(resume_from) / "optimizer.pt").exists():
                self.opt.load_state_dict(torch.load(Path(resume_from) / "optimizer.pt"))
        self.log_path = self.out / "log.jsonl"

    # ---------- rollouts ----------
    def rollout_group(self, browser: Browser, page: dict) -> list[dict]:
        self.model.eval()
        policy = TrainedPolicy(self.gen, sample=True, temperature=TEMPERATURE)
        recs = []
        for _ in range(G):
            rec = run_episode(policy, browser, page)
            rec["reward"] = reward(rec)
            rec["reward_components"] = components(rec)
            recs.append(rec)
        return recs

    # ---------- loss ----------
    def turn_loss(self, turn: dict, advantage: float) -> tuple[torch.Tensor, float, float]:
        inputs = encode_turn(self.processor, turn["messages"], turn["completion_ids"])
        n = len(turn["completion_ids"])
        with torch.no_grad():
            with self.model.disable_adapter():
                ref_lp = completion_logprobs(self.model, inputs, n, TEMPERATURE).detach()
        lp = completion_logprobs(self.model, inputs, n, TEMPERATURE)
        kl = torch.exp(ref_lp - lp) - (ref_lp - lp) - 1.0            # k3 estimator, per token
        loss = -advantage * lp.mean() + KL_BETA * kl.mean()
        return loss, float(lp.mean().item()), float(kl.mean().item())

    def update(self, browser: Browser) -> dict:
        t0 = time.time()
        batch = []
        for _ in range(self.P):
            if self.cursor >= len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            page = self.train_pages[self.order[self.cursor]]
            self.cursor += 1
            batch.append(page)
        groups = [self.rollout_group(browser, page) for page in batch]
        t_roll = time.time() - t0
        # advantages
        samples = []
        for group in groups:
            for rec, adv in zip(group, group_advantages([r["reward"] for r in group])):
                rec["advantage"] = adv
                if adv != 0.0:
                    for turn in rec["turns"]:
                        if turn["completion_ids"]:
                            samples.append((turn, adv))
        # optimisation: one update per batch
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        losses, lps, kls = [], [], []
        for turn, adv in samples:
            loss, lp, kl = self.turn_loss(turn, adv)
            (loss / max(1, len(samples))).backward()
            losses.append(loss.item()); lps.append(lp); kls.append(kl)
            del loss
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, GRAD_CLIP)) if samples else 0.0
        if samples:
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        self.step += 1
        flat = [r for g in groups for r in g]
        entry = {
            "step": self.step, "pages": [p["page_id"] for p in batch],
            "mean_reward": sum(r["reward"] for r in flat) / len(flat),
            "success_rate": sum(int(r["success"]) for r in flat) / len(flat),
            "mean_tool_calls": sum(r["tool_calls"] for r in flat) / len(flat),
            "mean_verify": sum(r["verify_calls"] for r in flat) / len(flat),
            "wrong_target_rate": sum(int(r["wrong_target_clicks"] > 0) for r in flat) / len(flat),
            "invalid_outputs": sum(sum(1 for a in r["actions"] if not a["valid"]) for r in flat),
            "n_samples": len(samples), "loss": (sum(losses) / len(losses)) if losses else None,
            "mean_logprob": (sum(lps) / len(lps)) if lps else None, "kl": (sum(kls) / len(kls)) if kls else None,
            "grad_norm": grad_norm if samples else None,
            "mean_turns_per_episode": sum(len(r["turns"]) for r in flat) / len(flat), "rollout_s": t_roll, "update_s": time.time() - t0 - t_roll,
            "total_s": time.time() - t0, "reward_std_zero_groups": sum(1 for g in groups if len({r["reward"] for r in g}) == 1),
            "mem_gb": torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else None,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        ep_dir = self.out / "episodes"
        ep_dir.mkdir(exist_ok=True)
        (ep_dir / f"step{self.step:04d}.json").write_text(json.dumps([strip_turns(r) for r in flat]))
        return entry

    # ---------- dev eval / checkpoints ----------
    def dev_eval(self, browser: Browser) -> dict:
        self.model.eval()
        policy = TrainedPolicy(self.gen, sample=False)
        recs = [run_episode(policy, browser, p) for p in self.dev_pages]
        for r in recs:
            r["reward"] = reward(r)
        res = {"step": self.step, "n": len(recs), "mean_reward": sum(r["reward"] for r in recs) / len(recs),
               "success_rate": sum(int(r["success"]) for r in recs) / len(recs),
               "mean_tool_calls": sum(r["tool_calls"] for r in recs) / len(recs),
               "wrong_target_rate": sum(int(r["wrong_target_clicks"] > 0) for r in recs) / len(recs),
               "invalid_outputs": sum(sum(1 for a in r["actions"] if not a["valid"]) for r in recs)}
        with open(self.out / "dev.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        (self.out / "dev_episodes").mkdir(exist_ok=True)
        (self.out / "dev_episodes" / f"step{self.step:04d}.json").write_text(json.dumps([strip_turns(r) for r in recs]))
        return res

    def save(self, name: str):
        d = self.out / name
        if d.exists():
            shutil.rmtree(d)
        self.model.save_pretrained(d)
        torch.save(self.opt.state_dict(), d / "optimizer.pt")
        (self.out / "state.json").write_text(json.dumps({"step": self.step, "cursor": self.cursor, "best": self.best,
                                                          "order": self.order}))

    def run(self):
        with Browser(headless=True) as browser:
            if self.step == 0:
                res = self.dev_eval(browser)   # LoRA B=0 at init: this is the base model's dev score
                print(json.dumps({"dev": res}), flush=True)
                self.best = {"step": 0, "mean_reward": res["mean_reward"]}
                self.save("best")   # base model (LoRA B=0) is the initial best; replaced when a dev eval ties or beats it
            while self.step < self.updates:
                entry = self.update(browser)
                print(json.dumps(entry), flush=True)
                if entry["loss"] is not None and not (math.isfinite(entry["loss"]) and math.isfinite(entry["kl"])):
                    print("non-finite loss; stopping", flush=True)
                    self.save("crashed")
                    return 2
                if self.step % self.checkpoint_every == 0:
                    self.save(f"ckpt_{self.step:04d}")
                    self.save("last")
                if self.step % self.dev_every == 0:
                    res = self.dev_eval(browser)
                    print(json.dumps({"dev": res}), flush=True)
                    if self.best is None or res["mean_reward"] >= self.best["mean_reward"]:
                        self.best = {"step": self.step, "mean_reward": res["mean_reward"]}
                        self.save("best")
                        (self.out / "state.json").write_text(json.dumps({"step": self.step, "cursor": self.cursor,
                                                                          "best": self.best, "order": self.order}))
            self.save("last")
        self.write_summary()
        return 0

    def write_summary(self):
        log = [json.loads(l) for l in self.log_path.read_text().splitlines()] if self.log_path.exists() else []
        dev = [json.loads(l) for l in (self.out / "dev.jsonl").read_text().splitlines()] if (self.out / "dev.jsonl").exists() else []
        summary = {"updates": len(log), "losses": [e["loss"] for e in log], "kls": [e["kl"] for e in log],
                   "n_samples": [e["n_samples"] for e in log], "grad_norms": [e["grad_norm"] for e in log],
                   "all_finite": all(e["loss"] is None or (math.isfinite(e["loss"]) and math.isfinite(e["kl"])) for e in log),
                   "updates_with_samples": sum(1 for e in log if e["n_samples"] > 0),
                   "mean_reward_per_update": [e["mean_reward"] for e in log], "dev": dev, "best": self.best,
                   "config": {"G": G, "temperature": TEMPERATURE, "lr": LR, "kl_beta": KL_BETA, "grad_clip": GRAD_CLIP,
                              "pages_per_update": self.P}}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/train")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--pages-per-update", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--dev-every", type=int, default=10)
    ap.add_argument("--dev-n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="adapter dir to resume from (e.g. runs/train/last)")
    args = ap.parse_args(argv)
    t = Trainer(Path(args.out), args.pages_per_update, args.updates, args.checkpoint_every, args.dev_every,
                args.dev_n, args.seed, args.resume)
    return t.run()


if __name__ == "__main__":
    sys.exit(main())
