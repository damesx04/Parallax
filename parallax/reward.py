"""Gated-aggregation reward.

    R = S * (1 + BETA * C) - LAMBDA * max(0, V - ANCHOR)

S = task success, C = consistency (only meaningful when S = 1), V = number of
verify_target calls. Computed from an episode record (parallax.env).
"""
from __future__ import annotations

BETA = 0.5
LAMBDA = 0.25
ANCHOR = 1
VISIBLE_THRESHOLD = 0.5


def _clickable(v_result: dict) -> bool:
    """The verification found the node clickable at its fresh centre."""
    return bool(v_result.get("hit_ok")) and float(v_result.get("visible_fraction", 0.0)) >= VISIBLE_THRESHOLD


def _found_obstruction(v_result: dict, intermediate_path: list[int], target_path: list[int]) -> bool:
    """The dismissed element is what the verification found in the way: the intermediate click's
    recipient chain intersects v.occluders or v.hit_path, ignoring elements that are also
    ancestors of the target (body, wrappers) since those are structure, not obstruction."""
    structural = set(target_path or [])
    seen = (set(v_result.get("occluders") or []) | set(v_result.get("hit_path") or [])) - structural
    return bool(seen & (set(intermediate_path or []) - structural))


def consistency(record: dict) -> int:
    """Consistency C, from the episode's ordered actions."""
    if not record.get("success"):
        return 0
    actions = [a for a in record["actions"] if a.get("valid", a.get("tool") is not None)]
    if not actions:
        return 0
    final = actions[-1]
    if final["tool"] not in ("click_element", "click_coord") or final["result"].get("outcome") != "success":
        return 0
    target_id = record["target_id"]
    n = final["args"]["node_id"] if final["tool"] == "click_element" else target_id
    v_index, v = None, None
    for i in range(len(actions) - 2, -1, -1):
        a = actions[i]
        if a["tool"] == "verify_target" and a["args"].get("node_id") == n:
            v_index, v = i, a
            break
    if v is None or not v["result"].get("found"):
        return 0
    vr = v["result"]
    if _clickable(vr) and final["tool"] == "click_element":
        return 1                                                   # (a) incl. stale-box refresh
    if not _clickable(vr) and final["tool"] == "click_coord":
        return 1                                                   # (b)
    if not _clickable(vr) and final["tool"] == "click_element":
        for a in actions[v_index + 1:-1]:                          # (c) dismissed what v found
            if a["tool"] in ("click_element", "click_coord") and a["result"].get("outcome") == "intermediate":
                if _found_obstruction(vr, a["result"].get("recipient_path") or [], record.get("target_path") or []):
                    return 1
        return 0
    return 0


def reward(record: dict) -> float:
    s = 1 if record.get("success") else 0
    c = consistency(record) if s else 0
    v = record.get("verify_calls", 0)
    return s * (1 + BETA * c) - LAMBDA * max(0, v - ANCHOR)


def components(record: dict) -> dict:
    s = 1 if record.get("success") else 0
    c = consistency(record) if s else 0
    v = record.get("verify_calls", 0)
    return {"S": s, "C": c, "V": v, "penalty": LAMBDA * max(0, v - ANCHOR), "R": reward(record)}
