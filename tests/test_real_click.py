"""Real clicks: every click is a real input event at screen coordinates."""
import re
from pathlib import Path

from parallax.browser import goto
from parallax.env import Episode

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]


def _episode(browser, target="#target"):
    ep = Episode(browser, str(FIX / "overlay.html"), "Click Place order", target)
    ep.reset()
    return ep


def _id(page, sel):
    return page.evaluate("s => window.__parallax.idFor(s)", sel)


def test_click_element_is_intercepted_by_overlay(browser):
    ep = _episode(browser)
    obs, done, info = ep.step({"tool": "click_element", "args": {"node_id": ep.target_id}})
    res = info["result"]
    assert res["clicked"] and res["is_trusted"] is True
    assert res["recipient"]["id"] == _id(ep.page, "#cover")
    assert res["outcome"] == "wrong_target" and done and ep.termination == "wrong_target"
    ep.close()


def test_click_coord_is_intercepted_by_overlay(browser):
    ep = _episode(browser)
    entry = [e for e in ep.obs["tree"] if e["id"] == ep.target_id][0]
    cx, cy = round(entry["box"][0] + entry["box"][2] / 2), round(entry["box"][1] + entry["box"][3] / 2)
    obs, done, info = ep.step({"tool": "click_coord", "args": {"x": cx, "y": cy}})
    res = info["result"]
    assert res["is_trusted"] is True
    assert res["recipient"]["id"] == _id(ep.page, "#cover")
    ep.close()


def test_click_coord_on_exposed_part_reaches_target(browser):
    ep = _episode(browser)
    obs, done, info = ep.step({"tool": "click_coord", "args": {"x": 240, "y": 120}})
    res = info["result"]
    assert res["is_trusted"] is True and res["outcome"] == "success" and ep.success
    ep.close()


def test_negative_control_js_click_is_not_trusted_and_bypasses_overlay(page):
    goto(page, str(FIX / "overlay.html"))
    page.evaluate("() => window.__parallax.clearClicks()")
    page.evaluate("() => document.getElementById('target').click()")  # what we must NOT do
    rec = page.evaluate("() => window.__parallax.clicks()")[-1]
    assert rec["isTrusted"] is False
    assert rec["id"] == _id(page, "#target")  # JS click ignores the overlay


def test_clicks_go_through_cdp_input_dispatch(browser):
    """Both click tools send Input.dispatchMouseEvent on the CDP session."""
    for which in ("click_element", "click_coord"):
        ep = _episode(browser)
        entry = [e for e in ep.obs["tree"] if e["id"] == ep.target_id][0]
        cx, cy = round(entry["box"][0] + entry["box"][2] / 2), round(entry["box"][1] + entry["box"][3] / 2)
        action = {"tool": "click_element", "args": {"node_id": ep.target_id}} if which == "click_element" \
            else {"tool": "click_coord", "args": {"x": 150, "y": 120}}
        sent = []
        real_send = ep.cdp.send

        def spy(method, params=None, _real=real_send):
            sent.append((method, params))
            return _real(method, params)

        ep.cdp.send = spy
        ep.step(action)
        mouse = [p for m, p in sent if m == "Input.dispatchMouseEvent"]
        assert [p["type"] for p in mouse] == ["mouseMoved", "mousePressed", "mouseReleased"], sent
        press = mouse[1]
        assert press["button"] == "left"
        assert press["clickCount"] == 1
        expected = (cx, cy) if which == "click_element" else (150, 120)
        assert (press["x"], press["y"]) == expected
        assert all((p["x"], p["y"]) == expected for p in mouse)
        ep.close()


ALLOWED_JS_CLICK_USES = ('addEventListener("click"', "addEventListener('click'")


def _strip_allowed(text: str) -> str:
    for a in ALLOWED_JS_CLICK_USES:
        text = text.replace(a, "")
    return text


def test_no_js_click_dispatch_anywhere_in_package():
    """No JS-dispatched clicks in any Python or JS text of the package (extension and page runtime included)."""
    files = list((ROOT / "parallax").rglob("*.py")) + list((ROOT / "parallax").rglob("*.js"))
    assert files
    joined = "\n".join(f.read_text() for f in files)
    assert "Input.dispatchMouseEvent" in joined
    assert "page.mouse.click(" not in joined
    for f in files:
        text = _strip_allowed(f.read_text())
        for needle in (".click(", "dispatchEvent(", "new MouseEvent", "new PointerEvent"):
            assert needle not in text, (f, needle)
