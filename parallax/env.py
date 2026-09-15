"""Episode environment: virtual time, the 3 tools, termination.

The environment, not a tool, advances time (ADVANCE_MS before every action),
enforces the step limit and ends the episode.
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Page

from parallax.browser import VIEWPORT, Browser, goto
from parallax.obs import build_observation
from parallax.tools import InvalidAction, TOOL_NAMES, validate_action
from parallax.tree import clear_overlay, draw_overlay, get_tree

MAX_STEPS = 4
ADVANCE_MS = 1000
VISIBLE_THRESHOLD = 0.5
BOX_TOLERANCE_PX = 2
CROP_W, CROP_H = 224, 168
PROBE_GRID = (5, 5)


def _crop_around(png: bytes, box) -> bytes:
    im = Image.open(io.BytesIO(png))
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    x0 = int(max(0, min(im.width - CROP_W, cx - CROP_W / 2)))
    y0 = int(max(0, min(im.height - CROP_H, cy - CROP_H / 2)))
    out = io.BytesIO()
    im.crop((x0, y0, x0 + CROP_W, y0 + CROP_H)).save(out, format="PNG")
    return out.getvalue()


def _clip(box):
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(VIEWPORT["width"], x + w), min(VIEWPORT["height"], y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _diff_count(a: bytes, b: bytes) -> int:
    ia = np.asarray(Image.open(io.BytesIO(a)).convert("RGB"), dtype=np.int16)
    ib = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.int16)
    if ia.shape != ib.shape:
        return 0
    return int((np.abs(ia - ib).max(axis=2) > 8).sum())


class Episode:
    """One page + one click task."""

    def __init__(self, browser: Browser, html_path: str, task: str, target_selector: str,
                 allowed_intermediate_selector: str | None = None, max_steps: int = MAX_STEPS,
                 meta: dict | None = None):
        self.browser = browser
        self.html_path = str(html_path)
        self.task = task
        self.target_selector = target_selector
        self.allowed_intermediate_selector = allowed_intermediate_selector
        self.max_steps = max_steps
        self.meta = meta or {}
        self.page: Page | None = None
        self.obs: dict | None = None
        self.history: list[dict] = []
        self.step_count = 0
        self.tool_calls = 0
        self.verify_calls = 0
        self.done = False
        self.success = False
        self.termination: str | None = None
        self.wrong_target_clicks = 0
        self.clicks: list[dict] = []
        self.actions: list[dict] = []
        self.target_id: int | None = None
        self.target_path: list[int] = []
        self.intermediate_id: int | None = None
        self.browser_version: str | None = None
        self.virtual_time = 0
        self.clock = None
        self._last_verify_crop: bytes | None = None
        self._last_verify_box = None
        self._last_kept: list[int] = []
        self._last_truncated = False
        self._last_marked_png: bytes | None = None
        self._last_tree: list[dict] | None = None
        self.cdp = None
        self._reuse_capture = False

    # ---------- lifecycle ----------
    def reset(self) -> dict:
        self.page = self.browser.new_page()
        goto(self.page, self.html_path)
        self.cdp = self.browser.context.new_cdp_session(self.page)
        self.target_id = self.page.evaluate("s => window.__parallax.idFor(s)", self.target_selector)
        if self.target_id is None:
            raise RuntimeError(f"target selector {self.target_selector!r} not found in {self.html_path}")
        self.target_path = self.page.evaluate("id => window.__parallax.pathOf(id)", self.target_id)
        self.browser_version = self.cdp.send("Browser.getVersion")["product"]
        if self.allowed_intermediate_selector:
            self.intermediate_id = self.page.evaluate("s => window.__parallax.idFor(s)", self.allowed_intermediate_selector)
        self.page.evaluate("() => window.__parallax.clearClicks()")
        self.obs = self._observe()
        return self.obs

    def close(self):
        if self.page is not None:
            try:
                self.page.close()
            except Exception:
                pass
            self.page = None

    def _advance_time(self, ms: int = ADVANCE_MS):
        """Advances the page's virtual clock (parallax_runtime.js). Pages without the
        runtime (static fixtures) record clock=False and no virtual time."""
        now = self.page.evaluate("ms => (window.__px ? window.__px.advance(ms) : null)", ms)
        if now is None:
            self.clock = False
            self.virtual_time = None
        else:
            self.clock = True
            self.virtual_time = now

    def _capture_marked(self) -> tuple[list[dict], bytes]:
        draw_overlay(self.page)
        tree = get_tree(self.page)
        png = self.page.screenshot(type="png")
        self._last_tree, self._last_marked_png = tree, png
        return tree, png

    def _observe(self, refresh: bool = True) -> dict:
        """Refresh rule: re-capture after executed tool calls only.
        A verify_target already captured the fresh marked state (reused here)."""
        fresh_capture = self._last_tree is None or refresh   # a verify's capture is fresh too (reused, not repeated)
        if self._last_tree is None or (refresh and not self._reuse_capture):
            self._capture_marked()
        self._reuse_capture = False
        if fresh_capture:
            obs = build_observation(task=self.task, tree=self._last_tree, screenshot_png=self._last_marked_png,
                                    history=list(self.history), verify_crop_png=self._last_verify_crop,
                                    step=self.step_count, max_steps=self.max_steps)
            self._last_kept = [e["id"] for e in obs["tree"]]
            self._last_truncated = obs["truncated"]
        else:
            # invalid output: the tree stays pinned to the last capture's kept entries (no re-truncation);
            # only the history note and step counter change
            kept = [e for e in self._last_tree if e["id"] in set(self._last_kept)]
            obs = build_observation(task=self.task, tree=kept, screenshot_png=self._last_marked_png,
                                    history=list(self.history), verify_crop_png=self._last_verify_crop,
                                    step=self.step_count, max_steps=self.max_steps, budget=10 ** 9)
            obs["truncated"] = self._last_truncated
        if obs["truncated"] and fresh_capture:
            # the marked screenshot must show exactly the ids the agent can act on (re-drawn once per capture)
            kept = [e["id"] for e in obs["tree"]]
            self.page.evaluate("ids => window.__parallax.overlay(true, ids)", kept)
            self._last_marked_png = self.page.screenshot(type="png")
            if self._last_verify_box is not None:   # keep the crop from the same capture as the screenshot
                self._last_verify_crop = _crop_around(self._last_marked_png, self._last_verify_box)
                obs["verify_crop_png"] = base64.b64encode(self._last_verify_crop).decode()
        obs["screenshot_png"] = base64.b64encode(self._last_marked_png).decode()
        self._last_verify_crop = None
        self._last_verify_box = None
        return obs

    # ---------- stepping ----------
    def step(self, action, allowed: tuple[str, ...] = TOOL_NAMES, raw_text: str | None = None,
             error: str | None = None) -> tuple[dict, bool, dict]:
        """One environment step. `raw_text`/`error` are recorded for policy outputs."""
        if self.done:
            raise RuntimeError("episode is done")
        self.step_count += 1
        self._advance_time()  # the environment, not the tool, lets time pass (every step)
        try:
            if action is None:
                raise InvalidAction(error or "no parsable tool call")
            act = validate_action(action, allowed)
        except InvalidAction as exc:
            result = {"error": error or str(exc)}
            self.history.append({"step": self.step_count, "tool": None, "args": {}, "result": result})
            self.actions.append({"step": self.step_count, "raw": raw_text if raw_text is not None else action,
                                 "valid": False, "tool": None, "args": {}, "result": result,
                                 "virtual_time": self.virtual_time})
            self._maybe_end_on_limit()
            self.obs = self._observe(refresh=False)
            return self.obs, self.done, {"result": result}

        self.tool_calls += 1
        tool, args = act["tool"], act["args"]
        if tool == "click_element":
            result = self._click_element(args["node_id"])
        elif tool == "click_coord":
            result = self._click_at(args["x"], args["y"], intended=None)
        else:
            self.verify_calls += 1
            result = self._verify(args["node_id"])
        self.history.append({"step": self.step_count, "tool": tool, "args": args, "result": result})
        self.actions.append({"step": self.step_count, "raw": raw_text if raw_text is not None else action,
                             "valid": True, "tool": tool, "args": args, "result": result,
                             "virtual_time": self.virtual_time})
        if not self.done:
            self._maybe_end_on_limit()
        # refresh only after a tool call that did something (a dispatched click or a performed
        # verification); no-op calls (unknown id, out-of-viewport pixel) refresh nothing
        acted = bool(result.get("clicked")) if tool != "verify_target" else bool(result.get("found"))
        self.obs = self._observe(refresh=acted)
        return self.obs, self.done, {"result": result}

    def _maybe_end_on_limit(self):
        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            self.termination = "step_limit"

    # ---------- tools ----------
    def _nearest_entry(self, path: list[int]):
        """Nearest ancestor-or-self in the current observation's tree."""
        for pid in path:
            e = self._entry(pid)
            if e is not None:
                return {"id": e["id"], "role": e["role"], "name": e["name"], "box": e["box"]}
        return None

    def _entry(self, node_id: int):
        for e in self.obs["tree"]:
            if e["id"] == node_id:
                return e
        return None

    def _click_element(self, node_id: int) -> dict:
        e = self._entry(node_id)
        if e is None:
            return {"clicked": False, "error": f"unknown node id {node_id}"}
        x, y, w, h = e["box"]
        return self._click_at(int(round(x + w / 2)), int(round(y + h / 2)), intended=node_id)

    def _click_at(self, x: int, y: int, intended: int | None) -> dict:
        if not (0 <= x < VIEWPORT["width"] and 0 <= y < VIEWPORT["height"]):
            return {"clicked": False, "error": "coordinates outside the viewport"}
        self.page.evaluate("() => window.__parallax.clearClicks()")
        self._dispatch_click(x, y)
        recorded = self.page.evaluate("() => window.__parallax.clicks()")
        # the first recorded event is the one Chrome delivered to the hit-tested element; a label may
        # forward a second activation click to its control, which is a consequence, not the recipient
        rec = recorded[0] if recorded else None
        recipient = None
        path = []
        if rec:
            path = rec.get("path") or []
            recipient = self._nearest_entry(path)
        result = {"clicked": True, "x": x, "y": y, "intended": intended, "recipient": recipient,
                  "recipient_path": path, "is_trusted": bool(rec and rec.get("isTrusted")), "outcome": None}
        if self.target_id in path:
            result["outcome"] = "success"
            self.success = True
            self.done = True
            self.termination = "success"
        elif self.intermediate_id is not None and self.intermediate_id in path:
            result["outcome"] = "intermediate"
        else:
            result["outcome"] = "wrong_target"
            self.wrong_target_clicks += 1
            self.done = True
            self.termination = "wrong_target"
        self.clicks.append({"step": self.step_count, **result})   # after the outcome is known
        return result

    def _dispatch_click(self, x: int, y: int) -> None:
        """Real input event at viewport CSS coordinates via CDP."""
        base = {"x": x, "y": y, "modifiers": 0}
        self.cdp.send("Input.dispatchMouseEvent", {**base, "type": "mouseMoved", "button": "none", "buttons": 0})
        self.cdp.send("Input.dispatchMouseEvent", {**base, "type": "mousePressed", "button": "left", "buttons": 1, "clickCount": 1})
        self.cdp.send("Input.dispatchMouseEvent", {**base, "type": "mouseReleased", "button": "left", "buttons": 0, "clickCount": 1})

    def _verify(self, node_id: int) -> dict:
        e = self._entry(node_id)
        if e is None:   # unknown to the current observation: found False
            return {"found": False, "node_id": node_id, "agrees": False, "clear_point": None}
        observed_box = e["box"]
        probe = self.page.evaluate("([id, nx, ny]) => window.__parallax.probe(id, nx, ny)", [node_id, *PROBE_GRID])
        if not probe.get("found"):
            return {"found": False, "node_id": node_id, "agrees": False, "clear_point": None}
        fresh = probe["box"]
        hit = probe["center"]
        hit_ok = node_id in (hit.get("path") or [])
        visible_fraction = self._visible_fraction(node_id, fresh, probe["occluders"])
        box_ok = observed_box is not None and all(abs(a - b) <= BOX_TOLERANCE_PX for a, b in zip(observed_box, fresh))
        agrees = bool(hit_ok and visible_fraction >= VISIBLE_THRESHOLD and box_ok)
        hit_desc = self._nearest_entry(hit.get("path") or [])
        # crop of the fresh marked screenshot; the same capture becomes the next observation
        _, png = self._capture_marked()
        self._reuse_capture = True
        self._last_verify_crop = _crop_around(png, fresh)
        self._last_verify_box = fresh
        return {"found": True, "node_id": node_id, "observed_box": observed_box, "fresh_box": fresh,
                "hit": hit_desc, "hit_id": hit.get("id"), "hit_path": hit.get("path") or [], "hit_ok": hit_ok, "box_ok": box_ok,
                "visible_fraction": round(visible_fraction, 3), "agrees": agrees,
                "clear_point": probe.get("clear_point"), "occluders": probe["occluders"],
                "role": probe.get("role"), "name": probe.get("name")}

    def _visible_fraction(self, node_id: int, box, occluders: list[int]) -> float:
        """Vision check: fraction of the pixels n paints that are actually visible."""
        clip = _clip(box)
        if clip is None:
            return 0.0
        clear_overlay(self.page)
        shot = lambda: self.page.screenshot(type="png", clip=clip)
        normal = shot()
        self.page.evaluate("id => window.__parallax.setHidden(id, true)", node_id)
        hidden = shot()
        self.page.evaluate("id => window.__parallax.setHidden(id, false)", node_id)
        visible_painted = _diff_count(normal, hidden)
        if occluders:
            self.page.evaluate("ids => window.__parallax.setHiddenMany(ids, true)", occluders)
            un_normal = shot()
            self.page.evaluate("id => window.__parallax.setHidden(id, true)", node_id)
            un_hidden = shot()
            self.page.evaluate("id => window.__parallax.setHidden(id, false)", node_id)
            self.page.evaluate("ids => window.__parallax.setHiddenMany(ids, false)", occluders)
            all_painted = _diff_count(un_normal, un_hidden)
        else:
            all_painted = visible_painted
        if all_painted == 0:
            return 0.0
        return min(1.0, visible_painted / all_painted)

    # ---------- record ----------
    def record(self) -> dict:
        m = self.meta
        return {
            "page_id": m.get("page_id"), "category": m.get("category"), "variant": m.get("variant"),
            "seed": m.get("seed"), "template": m.get("template"),
            "html_path": self.html_path, "task": self.task, "target_id": self.target_id,
            "target_path": self.target_path, "intermediate_id": self.intermediate_id, "meta": self.meta,
            "browser_version": self.browser_version,
            "steps": self.step_count, "tool_calls": self.tool_calls, "verify_calls": self.verify_calls,
            "success": self.success, "termination": self.termination,
            "wrong_target_clicks": self.wrong_target_clicks, "real_clicks": len(self.clicks),
            "clicks": self.clicks, "actions": self.actions,
            "virtual_time": self.virtual_time, "clock": self.clock, "history": self.history,
        }


def save_record(rec: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=1))
