"""Parse a policy's text output into a tool call."""
from __future__ import annotations

import json
import re

from parallax.tools import InvalidAction, TOOL_NAMES, validate_action

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str):
    """Returns the first JSON object in text, or raises InvalidAction."""
    if text is None:
        raise InvalidAction("empty output")
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise InvalidAction("no JSON object in output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as exc:
                    raise InvalidAction(f"malformed JSON: {exc}")
    raise InvalidAction("unbalanced JSON object")


def parse_action(text: str, allowed: tuple[str, ...] = TOOL_NAMES) -> dict:
    obj = extract_json(text)
    if isinstance(obj, dict) and "tool" in obj and "args" not in obj:
        # tolerate {"tool": "click_element", "node_id": 3}
        obj = {"tool": obj["tool"], "args": {k: v for k, v in obj.items() if k != "tool"}}
    if isinstance(obj, dict) and "name" in obj and "tool" not in obj:
        obj = {"tool": obj["name"], "args": obj.get("arguments", obj.get("args", {}))}
    return validate_action(obj, allowed)


def try_parse(text: str, allowed: tuple[str, ...] = TOOL_NAMES):
    """Returns (action | None, error | None)."""
    try:
        return parse_action(text, allowed), None
    except InvalidAction as exc:
        return None, str(exc)
