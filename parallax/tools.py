"""The 3-tool action interface. Execution lives in parallax.env.

Every click is a real input event: the environment sends CDP
Input.dispatchMouseEvent (mouseMoved, mousePressed, mouseReleased) on an
explicit CDP session at viewport CSS coordinates. Never a JS click.
"""
from __future__ import annotations

TOOLS = {
    "click_element": {"node_id": int},
    "click_coord": {"x": int, "y": int},
    "verify_target": {"node_id": int},
}
TOOL_NAMES = tuple(TOOLS)
assert len(TOOL_NAMES) == 3


class InvalidAction(ValueError):
    pass


def validate_action(action, allowed: tuple[str, ...] = TOOL_NAMES) -> dict:
    """Return {"tool": name, "args": {...}} with coerced ints, or raise InvalidAction."""
    if not isinstance(action, dict):
        raise InvalidAction("action must be an object")
    tool = action.get("tool")
    if tool not in TOOLS:
        raise InvalidAction(f"unknown tool {tool!r}")
    if tool not in allowed:
        raise InvalidAction(f"tool {tool!r} not available")
    args = action.get("args")
    if not isinstance(args, dict):
        raise InvalidAction("args must be an object")
    spec = TOOLS[tool]
    if set(args) != set(spec):
        raise InvalidAction(f"{tool} expects args {sorted(spec)}, got {sorted(args)}")
    out = {}
    for k, typ in spec.items():
        v = args[k]
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise InvalidAction(f"{k} must be a number")
        try:
            out[k] = int(round(float(v)))
        except (TypeError, ValueError):
            raise InvalidAction(f"{k} must be a number")
    return {"tool": tool, "args": out}
