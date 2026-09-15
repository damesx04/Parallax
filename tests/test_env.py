"""Environment: verify_target semantics, virtual time, termination."""
from pathlib import Path

from parallax.env import ADVANCE_MS, MAX_STEPS, Episode

FIX = Path(__file__).parent / "fixtures"


def _id(page, sel):
    return page.evaluate("s => window.__parallax.idFor(s)", sel)


def test_verify_agrees_on_clean_target(browser):
    ep = Episode(browser, str(FIX / "overlay.html"), "Click Cancel order", "#clean")
    ep.reset()
    _, done, info = ep.step({"tool": "verify_target", "args": {"node_id": ep.target_id}})
    r = info["result"]
    assert r["found"] and r["agrees"] is True and r["hit_ok"] and r["box_ok"]
    assert r["visible_fraction"] >= 0.5
    assert r["clear_point"] is not None and not done
    assert ep.verify_calls == 1 and ep.tool_calls == 1
    # the crop is delivered in the next observation and only there
    assert ep.obs["verify_crop_png"] is not None
    ep.close()


def test_verify_disagrees_under_partial_cover_with_clear_point(browser):
    ep = Episode(browser, str(FIX / "overlay.html"), "Click Place order", "#target")
    ep.reset()
    _, done, info = ep.step({"tool": "verify_target", "args": {"node_id": ep.target_id}})
    r = info["result"]
    assert r["agrees"] is False and r["hit_ok"] is False
    assert r["hit"]["id"] == _id(ep.page, "#cover")
    assert r["hit_path"][0] == _id(ep.page, "#cover") and ep.target_id not in r["hit_path"]
    assert ep.page.evaluate("() => window.__parallax.idOf(document.body)") in r["hit_path"]
    assert _id(ep.page, "#cover") in r["occluders"]
    assert 0.0 < r["visible_fraction"] < 1.0
    cp = r["clear_point"]
    assert cp is not None and 100 <= cp[0] <= 260 and 100 <= cp[1] <= 140
    # acting on the clear point succeeds
    _, done, info = ep.step({"tool": "click_coord", "args": {"x": cp[0], "y": cp[1]}})
    assert info["result"]["outcome"] == "success" and done and ep.success
    ep.close()


def test_verify_full_cover_no_clear_point(browser):
    ep = Episode(browser, str(FIX / "overlay.html"), "Click Hidden one", "#full")
    ep.reset()
    _, _, info = ep.step({"tool": "verify_target", "args": {"node_id": ep.target_id}})
    r = info["result"]
    assert r["agrees"] is False and r["clear_point"] is None
    assert r["visible_fraction"] < 0.05
    ep.close()


def test_verify_disagrees_after_scheduled_reflow_and_refreshes(browser):
    ep = Episode(browser, str(FIX / "reflow.html"), "Click Save draft", "#first")
    obs0 = ep.reset()
    box0 = [e for e in obs0["tree"] if e["id"] == ep.target_id][0]["box"]
    assert ep.page.evaluate("() => __px.now()") == 0
    _, done, info = ep.step({"tool": "verify_target", "args": {"node_id": ep.target_id}})
    r = info["result"]
    assert ep.page.evaluate("() => __px.now()") == ADVANCE_MS  # time advanced by the env, once
    assert r["box_ok"] is False and r["agrees"] is False
    assert r["fresh_box"][1] >= box0[1] + 40
    # the next observation is post-reflow, so click_element now lands correctly
    box1 = [e for e in ep.obs["tree"] if e["id"] == ep.target_id][0]["box"]
    assert box1 == r["fresh_box"]
    _, done, info = ep.step({"tool": "click_element", "args": {"node_id": ep.target_id}})
    assert info["result"]["outcome"] == "success" and ep.success
    assert ep.page.evaluate("() => __px.now()") == 2 * ADVANCE_MS
    ep.close()


def test_dom_only_click_is_fooled_by_reflow(browser):
    ep = Episode(browser, str(FIX / "reflow.html"), "Click Publish", "#second")
    ep.reset()
    _, done, info = ep.step({"tool": "click_element", "args": {"node_id": ep.target_id}})
    assert info["result"]["outcome"] == "wrong_target" and done
    assert info["result"]["recipient"]["id"] == _id(ep.page, "#first")
    ep.close()


def test_unknown_node_consumes_step_without_click_but_counts_as_tool_call(browser):
    """A validly parsed call counts even when it performs no click."""
    ep = Episode(browser, str(FIX / "overlay.html"), "Click Cancel order", "#clean")
    ep.reset()
    _, done, info = ep.step({"tool": "click_element", "args": {"node_id": 99999}})
    assert info["result"]["clicked"] is False and not done
    assert ep.step_count == 1 and ep.tool_calls == 1 and ep.clicks == []
    _, done, info = ep.step({"tool": "click_coord", "args": {"x": -5, "y": 10}})
    assert info["result"]["clicked"] is False and ep.tool_calls == 2 and ep.clicks == []
    _, done, info = ep.step({"tool": "verify_target", "args": {"node_id": 99999}})
    assert info["result"]["found"] is False and ep.tool_calls == 3 and ep.verify_calls == 1
    assert ep.record()["real_clicks"] == 0 and ep.record()["clock"] is False  # static fixture: no runtime clock
    ep.close()


def test_virtual_clock_equals_1000ms_per_step(browser):
    ep = Episode(browser, str(FIX / "reflow.html"), "Click Save draft", "#first")
    ep.reset()
    for i in range(1, 4):
        ep.step({"tool": "verify_target", "args": {"node_id": ep.target_id}} if i < 3 else {"tool": "nope", "args": {}})
        assert ep.page.evaluate("() => __px.now()") == ADVANCE_MS * i == ep.virtual_time
    assert ep.record()["clock"] is True
    ep.close()


def test_invalid_action_consumes_step_advances_time_no_refresh(browser):
    ep = Episode(browser, str(FIX / "reflow.html"), "Click Publish", "#second")
    obs0 = ep.reset()
    obs1, done, info = ep.step({"tool": "scroll", "args": {}})
    assert ep.step_count == 1 and ep.tool_calls == 0 and not done
    assert ep.page.evaluate("() => __px.now()") == ADVANCE_MS  # time passed while the agent produced garbage
    assert ep.history[-1]["tool"] is None and ep.actions[-1]["valid"] is False
    # no refresh: same tree and screenshot as before, so the stale boxes remain
    assert obs1["tree"] == obs0["tree"] and obs1["screenshot_png"] == obs0["screenshot_png"]
    assert obs1["step"] == 1 and len(obs1["history"]) == 1
    # and the stale click_element is still fooled (no free refresh from an invalid output)
    _, done, info = ep.step({"tool": "click_element", "args": {"node_id": ep.target_id}})
    assert info["result"]["outcome"] == "wrong_target"
    ep.close()


def test_verify_agrees_on_transparent_text_link(browser):
    ep = Episode(browser, str(FIX / "overlay.html"), "Click Return policy", "#lnk")
    ep.reset()
    _, _, info = ep.step({"tool": "verify_target", "args": {"node_id": ep.target_id}})
    r = info["result"]
    assert r["agrees"] is True and r["visible_fraction"] >= 0.95, r
    ep.close()


def test_recipient_reported_as_nearest_tree_entry(browser):
    """The recipient is mapped to the nearest ancestor-or-self that is a tree entry, else nothing."""
    ep = Episode(browser, str(FIX / "static.html"), "Click Place order", "#place")
    ep.reset()
    # the checkbox input sits inside a <label>; the tree entry is the checkbox itself, the label is not an entry,
    # so click the label's text (outside the input box): the recipient chain is label > div.row > ... > body
    x, y = ep.page.evaluate("() => { const r = document.querySelector('label:has(#gift)').getBoundingClientRect(); return [Math.round(r.right - 8), Math.round(r.top + r.height / 2)]; }")
    hit = ep.page.evaluate("([x, y]) => document.elementFromPoint(x, y).tagName", [x, y])
    assert hit == "LABEL"
    _, done, info = ep.step({"tool": "click_coord", "args": {"x": x, "y": y}})
    rec = info["result"]["recipient"]
    entry_ids = {e["id"] for e in ep.obs["tree"]} | {e["id"] for e in ep.obs["tree"]}
    assert rec is None or rec["id"] in entry_ids
    assert info["result"]["recipient_path"][0] == ep.page.evaluate("() => window.__parallax.idOf(document.querySelector('label:has(#gift)'))")
    ep.close()
    # background click: no tree entry in the chain -> nothing, and it is a wrong-target click
    ep = Episode(browser, str(FIX / "overlay.html"), "Click Cancel order", "#clean")
    ep.reset()
    _, done, info = ep.step({"tool": "click_coord", "args": {"x": 850, "y": 600}})
    assert info["result"]["recipient"] is None and info["result"]["outcome"] == "wrong_target" and done
    ep.close()


def test_truncated_page_budget_and_refresh_rule(browser):
    """A deliberately huge page is truncated under budget through a real Episode; the truncated
    screenshot shows only kept ids; an invalid output re-captures nothing even when truncated."""
    import base64, io
    from PIL import Image
    from parallax.obs import TEXT_TOKEN_BUDGET
    ep = Episode(browser, str(FIX / "huge.html"), "Click Publish", "#second")
    obs0 = ep.reset()
    assert obs0["truncated"] is True and obs0["token_count"] <= TEXT_TOKEN_BUDGET
    kept = {e["id"] for e in obs0["tree"]}
    assert ep.target_id in kept
    labels = {l["id"] for l in ep.page.evaluate("() => window.__parallax.labelBoxes()")}
    assert labels == kept  # overlay redrawn for exactly the kept entries
    obs1, done, info = ep.step({"tool": "nonsense " * 40, "args": {}})   # a long invalid note
    assert obs1["screenshot_png"] == obs0["screenshot_png"] and obs1["tree"] == obs0["tree"]
    assert obs1["truncated"] is True and obs1["tree_text"] == obs0["tree_text"]
    # the reflow (500 ms) has happened, but the stale observation is still what click_element uses
    _, done, info = ep.step({"tool": "click_element", "args": {"node_id": ep.target_id}})
    assert info["result"]["outcome"] == "wrong_target"
    ep.close()
