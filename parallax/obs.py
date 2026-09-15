"""Per-turn observation builder. Shared by every policy/baseline."""
from __future__ import annotations

import base64
import functools
import json
from pathlib import Path

from parallax.tree import render_entry

TEXT_TOKEN_BUDGET = 2500
SYSTEM_PROMPT_ALLOWANCE = 600   # fixed allowance for any policy's system prompt (asserted >= each real prompt)
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SCHEMA_PATH = Path(__file__).resolve().parent / "obs_schema.json"

# Lower number = dropped first when over budget.
_PRIORITY = {"paragraph": 0, "generic": 0, "label": 0, "region": 0, "article": 0,
             "img": 1, "listitem": 1, "list": 1, "heading": 2}


@functools.lru_cache(maxsize=1)
def tokenizer():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID)


def count_tokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])


def render_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    for h in history:
        tool, args, res = h["tool"], h["args"], h["result"]
        if tool is None:
            lines.append(f"{h['step']}. invalid output -> ignored ({res.get('error', '')})")
            continue
        a = ", ".join(f"{k}={v}" for k, v in args.items())
        if tool == "verify_target":
            if not res.get("found"):
                lines.append(f"{h['step']}. verify_target({a}) -> node not found")
            else:
                hit = res.get("hit")
                hit_s = f"[{hit['id']}] {hit['role']} \"{hit['name']}\"" if hit and hit.get("id") else "nothing"
                cp = res.get("clear_point")
                cp_s = f"({cp[0]},{cp[1]})" if cp else "none"
                fb = res.get("fresh_box")
                lines.append(
                    f"{h['step']}. verify_target({a}) -> agrees={str(res['agrees']).lower()}; "
                    f"now at ({fb[0]},{fb[1]},{fb[2]},{fb[3]}); at its centre: {hit_s}; "
                    f"visible={res['visible_fraction']:.2f}; clear_point={cp_s}"
                )
        else:
            if res.get("clicked"):
                rec = res.get("recipient")
                rec_s = f"[{rec['id']}] {rec['role']} \"{rec['name']}\"" if rec and rec.get("id") else "nothing"
                lines.append(f"{h['step']}. {tool}({a}) -> clicked at ({res['x']},{res['y']}); landed on {rec_s}")
            else:
                lines.append(f"{h['step']}. {tool}({a}) -> no click ({res.get('error', '')})")
    return "\n".join(lines)


def render_user_text(task: str, tree_text: str, history: list[dict], step: int, max_steps: int) -> str:
    return (
        f"Task: {task}\n\n"
        f"Accessibility tree ([id] role \"name\" [states] (x,y,w,h)):\n{tree_text}\n\n"
        f"Action history:\n{render_history(history)}\n\n"
        f"Steps used: {step} of {max_steps} (every output counts, even an unparsable one). Respond with one JSON tool call."
    )


def _fits(task: str, entries: list[dict], history: list[dict], step: int, max_steps: int):
    """Token count of the policy-independent text plus the fixed system-prompt allowance."""
    tree_text = "\n".join(render_entry(e) for e in entries)
    text = render_user_text(task, tree_text, history, step, max_steps)
    return SYSTEM_PROMPT_ALLOWANCE + count_tokens(text), tree_text


def build_observation(*, task: str, tree: list[dict], screenshot_png: bytes, history: list[dict],
                      verify_crop_png: bytes | None, step: int, max_steps: int,
                      budget: int = TEXT_TOKEN_BUDGET) -> dict:
    """Policy-independent: the same page/step yields identical bytes for every policy."""
    entries = list(tree)
    n, tree_text = _fits(task, entries, history, step, max_steps)
    truncated = False
    while n > budget and entries:
        truncated = True
        # drop the lowest-priority entry; among equals, the one furthest down the page
        idx = min(range(len(entries)), key=lambda i: (_PRIORITY.get(entries[i]["role"], 3), -entries[i]["box"][1]))
        entries.pop(idx)
        n, tree_text = _fits(task, entries, history, step, max_steps)
    if n > budget:
        # history alone exceeds the budget (impossible with 4 steps): keep the guarantee explicit
        raise RuntimeError(f"observation text exceeds the token budget even with an empty tree ({n} > {budget})")
    return {
        "task": task,
        "tree_text": tree_text,
        "tree": entries,
        "screenshot_png": base64.b64encode(screenshot_png).decode(),
        "history": history,
        "verify_crop_png": base64.b64encode(verify_crop_png).decode() if verify_crop_png else None,
        "step": step,
        "max_steps": max_steps,
        "token_count": n,
        "truncated": truncated,
    }


def validate(obs: dict) -> None:
    import jsonschema
    jsonschema.validate(obs, json.loads(SCHEMA_PATH.read_text()))
