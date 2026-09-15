"""Policies: policy scaffolds share one observation builder; A/B constraints hold."""
import json
from pathlib import Path


from parallax.model.policies import BaselineA, BaselineB, TrainedPolicy, run_episode
from parallax.pages.loader import DEV_DIR

ROOT = Path(__file__).resolve().parents[1]


class StubGenerator:
    """Deterministic fake model: answers with a scripted sequence of texts."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate(self, messages, sample, temperature=1.0):
        self.calls.append({"messages": messages, "sample": sample})
        text = self.script.pop(0) if self.script else "no idea"
        return {"text": text, "completion_ids": [1, 2, 3], "prompt_len": 10, "messages": messages}


def _page(pid="d0000"):
    m = json.loads((DEV_DIR / pid / "manifest.json").read_text())
    m["html_path"] = str(DEV_DIR / pid / "page.html")
    return m


def _has_image(messages):
    return any(c.get("type") == "image" for m in messages for c in m["content"])


def _target_id(rec):
    return rec["target_id"]


def test_baseline_a_no_image_and_click_element_only(browser):
    page = _page()
    gen = StubGenerator(['{"tool": "verify_target", "args": {"node_id": 5}}',
                        '{"tool": "click_coord", "args": {"x": 1, "y": 1}}',
                        '{"tool": "click_element", "args": {"node_id": 99999}}',
                        'garbage'])
    rec = run_episode(BaselineA(gen), browser, page)
    assert all(not _has_image(c["messages"]) for c in gen.calls)
    assert all(c["messages"][0]["content"][0]["text"].startswith("You are a web agent") for c in gen.calls)
    tools = [a["tool"] for a in rec["actions"]]
    assert set(tools) <= {"click_element", None}
    assert rec["verify_calls"] == 0 and rec["termination"] == "step_limit" and rec["steps"] == 4
    # disallowed tools were recorded as invalid steps, not executed
    assert [a["valid"] for a in rec["actions"]] == [False, False, True, False]


def test_baseline_b_verifies_before_every_click(browser):
    page = _page()  # decoy v1: verify disagrees, so B clicks something
    class Gen(StubGenerator):
        """Always intends/clicks the target (naive: ignores the verification result)."""

        def generate(self, messages, sample, temperature=1.0):
            self.calls.append({"messages": messages, "sample": sample})
            return {"text": '{"tool": "click_element", "args": {"node_id": %d}}' % self.target, "completion_ids": [1], "prompt_len": 1, "messages": messages}

    from parallax.env import Episode
    ep = Episode(browser, page["html_path"], page["task"], page["target_selector"])
    ep.reset(); target = ep.target_id; ep.close()
    g = Gen([]); g.target = target
    rec = run_episode(BaselineB(g), browser, page)
    tools = [a["tool"] for a in rec["actions"]]
    assert tools[0] == "verify_target" and tools[1] == "click_element"
    assert rec["tool_calls"] == 2 and rec["verify_calls"] == 1
    assert rec["termination"] == "wrong_target"  # naive: verified false, clicked the covered centre anyway
    assert all(_has_image(c["messages"]) for c in g.calls)
    assert "Step 1 of 2" in g.calls[0]["messages"][-1]["content"][-1]["text"]
    assert "Step 2 of 2" in g.calls[1]["messages"][-1]["content"][-1]["text"]


def test_baseline_b_invalid_intent_is_invalid_step(browser):
    rec = run_episode(BaselineB(StubGenerator(["???"] * 4)), browser, _page())
    assert rec["tool_calls"] == 0 and rec["steps"] == 4 and rec["termination"] == "step_limit"


def test_trained_policy_free_tool_choice(browser):
    page = _page()
    target, _ = _ids(browser, page)
    gen = StubGenerator(['{"tool": "verify_target", "args": {"node_id": %d}}' % target,
                         '{"tool": "click_coord", "args": {"x": 5, "y": 5}}'])
    rec = run_episode(TrainedPolicy(gen), browser, page)
    assert [a["tool"] for a in rec["actions"]] == ["verify_target", "click_coord"]
    assert _has_image(gen.calls[0]["messages"])
    assert len(rec["turns"]) == 2 and rec["turns"][0]["completion_ids"] == [1, 2, 3]
    # the verify crop is delivered as a second image on the following turn
    imgs = [c for c in gen.calls[1]["messages"][-1]["content"] if c.get("type") == "image"]
    assert len(imgs) == 2


def test_same_observation_builder_across_policies(browser):
    """Identical tree/screenshot/token_count for the same page and step."""
    page = _page("d0003")
    seen = []

    class Capture(StubGenerator):
        def generate(self, messages, sample, temperature=1.0):
            seen.append(messages)
            return {"text": "x", "completion_ids": [1], "prompt_len": 1, "messages": messages}

    from parallax.env import Episode
    obs = {}
    for name, cls in (("trained", TrainedPolicy), ("baseline_a", BaselineA), ("baseline_b", BaselineB)):
        ep = Episode(browser, page["html_path"], page["task"], page["target_selector"],
                     page.get("allowed_intermediate_selector"), meta=page)
        obs[name] = ep.reset(); ep.close()
    for k in ("tree_text", "tree", "screenshot_png", "token_count"):
        assert obs["trained"][k] == obs["baseline_a"][k] == obs["baseline_b"][k], k
    # and the user text is the same template, differing only by images / the B suffix
    from parallax.model.generate import build_messages
    ta = build_messages(obs["baseline_a"], BaselineA.system_prompt, False)[-1]["content"]
    tt = build_messages(obs["trained"], TrainedPolicy.system_prompt, True)[-1]["content"]
    assert ta[-1]["text"] == tt[-1]["text"]
    assert not _has_image([{"content": ta}]) and _has_image([{"content": tt}])


def _ids(browser, page):
    from parallax.env import Episode
    ep = Episode(browser, page["html_path"], page["task"], page["target_selector"], page.get("allowed_intermediate_selector"))
    ep.reset()
    ids = (ep.target_id, ep.intermediate_id)
    ep.close()
    return ids


def _dev_page_with_dismiss():
    from parallax.pages.loader import load_training_pages
    return [p for p in load_training_pages(DEV_DIR) if p["variant"] == 2 and p["category"] in ("decoy", "occluded")][0]


def test_baseline_b_two_cycles_on_dismiss_page(browser):
    """Worked example: verify, click dismiss, verify, click target = 4 tool calls."""
    page = _dev_page_with_dismiss()
    target, dismiss = _ids(browser, page)

    class Gen(StubGenerator):
        def generate(self, messages, sample, temperature=1.0):
            self.calls.append({"messages": messages, "sample": sample})
            user = messages[-1]["content"][-1]["text"]
            if "Step 1 of 2" in user:
                text = '{"tool": "click_element", "args": {"node_id": %d}}' % target
            else:
                clicked_dismiss = "intermediate" in user or ("landed on" in user and "clicked" in user)
                text = '{"tool": "click_element", "args": {"node_id": %d}}' % (target if clicked_dismiss else dismiss)
            return {"text": text, "completion_ids": [1], "prompt_len": 1, "messages": messages}

    rec = run_episode(BaselineB(Gen([])), browser, page)
    tools = [(a["tool"], a["args"].get("node_id")) for a in rec["actions"]]
    assert tools == [("verify_target", target), ("click_element", dismiss), ("verify_target", target), ("click_element", target)], tools
    assert rec["tool_calls"] == 4 and rec["verify_calls"] == 2 and rec["steps"] == 4
    assert rec["success"] and rec["termination"] == "success"
    assert rec["clicks"][0]["outcome"] == "intermediate"


def test_consistency_on_real_dismiss_episode(browser):
    """Reward clause (c) on a real episode: verify (finds the obstruction), dismiss, click target -> C=1."""
    from parallax.env import Episode
    from parallax.reward import consistency, reward
    page = _dev_page_with_dismiss()
    ep = Episode(browser, page["html_path"], page["task"], page["target_selector"], page.get("allowed_intermediate_selector"), meta=page)
    ep.reset()
    _, _, info = ep.step({"tool": "verify_target", "args": {"node_id": ep.target_id}})
    assert info["result"]["agrees"] is False and info["result"]["hit_ok"] is False
    ep.step({"tool": "click_element", "args": {"node_id": ep.intermediate_id}})
    ep.step({"tool": "click_element", "args": {"node_id": ep.target_id}})
    rec = ep.record(); ep.close()
    assert rec["success"] and consistency(rec) == 1 and reward(rec) == 1.5
    # the structural chain shared with the target (body, wrappers) is not what made it match:
    assert rec["target_path"] and set(rec["target_path"]) & set(rec["clicks"][0]["recipient_path"])
    # negative control on the same real record: with the structural ids removed from the verification's
    # finding, the same dismissal no longer matches -> C = 0
    import copy
    neg = copy.deepcopy(rec)
    v = neg["actions"][0]["result"]
    structural = set(rec["target_path"])
    v["occluders"] = [i for i in v["occluders"] if i in structural]
    v["hit_path"] = [i for i in v["hit_path"] if i in structural]
    assert consistency(neg) == 0 and reward(neg) == 1.0
