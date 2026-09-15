"""Scripted oracles used to validate attacks."""
from __future__ import annotations

from parallax.browser import Browser
from parallax.env import Episode


def make_episode(browser: Browser, page: dict) -> Episode:
    return Episode(browser, page["html_path"], page["task"], page["target_selector"],
                   page.get("allowed_intermediate_selector"), meta=page)


def dom_only(browser: Browser, page: dict) -> dict:
    """Clicks the true target by ID with no verification."""
    ep = make_episode(browser, page)
    try:
        ep.reset()
        ep.step({"tool": "click_element", "args": {"node_id": ep.target_id}})
        return ep.record()
    finally:
        ep.close()


def verify_then_act(browser: Browser, page: dict) -> dict:
    """verify; click_element if agrees; click_coord at clear_point; else dismiss then click_element."""
    ep = make_episode(browser, page)
    try:
        ep.reset()
        _, done, info = ep.step({"tool": "verify_target", "args": {"node_id": ep.target_id}})
        r = info["result"]
        if r.get("agrees"):
            ep.step({"tool": "click_element", "args": {"node_id": ep.target_id}})
        elif r.get("clear_point"):
            x, y = r["clear_point"]
            ep.step({"tool": "click_coord", "args": {"x": x, "y": y}})
        elif ep.intermediate_id is not None:
            _, done, _ = ep.step({"tool": "click_element", "args": {"node_id": ep.intermediate_id}})
            if not done:
                ep.step({"tool": "click_element", "args": {"node_id": ep.target_id}})
        else:
            ep.step({"tool": "click_element", "args": {"node_id": ep.target_id}})
        return ep.record()
    finally:
        ep.close()
