"""The trained policy and the two baselines, sharing one env,
one observation builder, one prompt template and one base model."""
from __future__ import annotations

from parallax.env import Episode
from parallax.model.actions import try_parse
from parallax.model.prompts import BASELINE_B_CLICK_SUFFIX, BASELINE_B_INTENT_SUFFIX, DOM_ONLY_SYSTEM, FULL_SYSTEM
from parallax.model.generate import build_messages
from parallax.tools import TOOL_NAMES


class Policy:
    name = "policy"
    system_prompt = FULL_SYSTEM
    with_image = True
    allowed = TOOL_NAMES

    def __init__(self, generator, sample: bool = False, temperature: float = 1.0):
        self.gen = generator
        self.sample = sample
        self.temperature = temperature

    def decide(self, obs: dict, allowed=None, extra: str | None = None) -> dict:
        allowed = allowed or self.allowed
        messages = build_messages(obs, self.system_prompt, self.with_image, extra)
        g = self.gen.generate(messages, sample=self.sample, temperature=self.temperature)
        action, err = try_parse(g["text"], allowed)
        g.update({"action": action, "error": err, "allowed": allowed})
        return g

    def step(self, ep: Episode) -> list[dict]:
        """One decision -> one env step. Returns the samples (for training)."""
        g = self.decide(ep.obs)
        ep.step(g["action"], allowed=self.allowed, raw_text=g["text"], error=g["error"])
        return [g]


class TrainedPolicy(Policy):
    name = "trained"


class BaselineA(Policy):
    """DOM-only: no image, click_element only."""
    name = "baseline_a"
    system_prompt = DOM_ONLY_SYSTEM
    with_image = False
    allowed = ("click_element",)


class BaselineB(Policy):
    """Naive verify-every-step: intent query -> forced verify_target -> click."""
    name = "baseline_b"

    def step(self, ep: Episode) -> list[dict]:
        samples = []
        g = self.decide(ep.obs, allowed=("click_element",), extra=BASELINE_B_INTENT_SUFFIX)
        samples.append(g)
        if g["action"] is None:
            ep.step(None, raw_text=g["text"], error=g["error"])  # invalid intent = invalid step
            return samples
        node_id = g["action"]["args"]["node_id"]
        ep.step({"tool": "verify_target", "args": {"node_id": node_id}}, raw_text=g["text"])
        if ep.done:
            return samples
        g2 = self.decide(ep.obs, allowed=("click_element", "click_coord"), extra=BASELINE_B_CLICK_SUFFIX)
        samples.append(g2)
        ep.step(g2["action"], allowed=("click_element", "click_coord"), raw_text=g2["text"], error=g2["error"])
        return samples


POLICIES = {"trained": TrainedPolicy, "baseline_a": BaselineA, "baseline_b": BaselineB}


def run_episode(policy: Policy, browser, page: dict, adapter_note: str | None = None) -> dict:
    """Runs one full episode; returns the record with per-turn samples attached."""
    ep = Episode(browser, page["html_path"], page["task"], page["target_selector"],
                 page.get("allowed_intermediate_selector"), meta=page)
    samples = []
    try:
        ep.reset()
        while not ep.done:
            samples.extend(policy.step(ep))
        rec = ep.record()
    finally:
        ep.close()
    rec["policy"] = policy.name
    rec["turns"] = [{"messages": s["messages"], "completion_ids": s["completion_ids"], "prompt_len": s["prompt_len"],
                     "text": s["text"], "action": s["action"], "error": s["error"], "allowed": list(s["allowed"])}
                    for s in samples]
    return rec
