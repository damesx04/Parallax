"""Observation: observation schema, token budget, screenshot size."""
import base64
import io
from pathlib import Path

from PIL import Image

from parallax.browser import VIEWPORT
from parallax.env import Episode
from parallax.model.prompts import BASELINE_B_CLICK_SUFFIX, BASELINE_B_INTENT_SUFFIX, DOM_ONLY_SYSTEM, FULL_SYSTEM
from parallax.obs import SYSTEM_PROMPT_ALLOWANCE, TEXT_TOKEN_BUDGET, build_observation, count_tokens, validate

FIX = Path(__file__).parent / "fixtures"


def test_observation_validates_and_screenshot_size(browser):
    ep = Episode(browser, str(FIX / "static.html"), "Click Place order", "#place")
    obs = ep.reset()
    validate(obs)
    im = Image.open(io.BytesIO(base64.b64decode(obs["screenshot_png"])))
    assert im.size == (VIEWPORT["width"], VIEWPORT["height"])
    assert obs["token_count"] <= TEXT_TOKEN_BUDGET and obs["truncated"] is False
    assert obs["step"] == 0 and obs["max_steps"] == 4 and obs["history"] == []
    assert "[%d] button \"Place order\"" % ep.target_id in obs["tree_text"]
    ep.step({"tool": "verify_target", "args": {"node_id": ep.target_id}})
    validate(ep.obs)
    assert ep.obs["history"][0]["tool"] == "verify_target"
    assert ep.obs["verify_crop_png"] is not None
    crop = Image.open(io.BytesIO(base64.b64decode(ep.obs["verify_crop_png"])))
    assert crop.size == (224, 168)
    ep.close()


def test_token_budget_truncates_huge_tree():
    tree = []
    for i in range(1, 1200):
        role = "paragraph" if i % 3 else "button"
        tree.append({"id": i, "role": role, "name": f"Entry number {i} with some longer text", "states": [],
                     "box": [10, i * 3, 100, 20], "depth": 0})
    obs = build_observation(task="Click Entry number 3", tree=tree, screenshot_png=b"png", history=[],
                            verify_crop_png=None, step=0, max_steps=4)
    assert obs["truncated"] is True
    assert obs["token_count"] <= TEXT_TOKEN_BUDGET
    kept_roles = {e["role"] for e in obs["tree"]}
    assert "button" in kept_roles  # interactive entries survive longest


def test_system_prompt_allowance_covers_every_policy_prompt():
    """The budget uses a fixed allowance so the tree shown is policy-independent."""
    assert count_tokens(FULL_SYSTEM) <= SYSTEM_PROMPT_ALLOWANCE
    assert count_tokens(DOM_ONLY_SYSTEM) <= SYSTEM_PROMPT_ALLOWANCE
    assert count_tokens(FULL_SYSTEM) + max(count_tokens(BASELINE_B_INTENT_SUFFIX), count_tokens(BASELINE_B_CLICK_SUFFIX)) <= SYSTEM_PROMPT_ALLOWANCE
    assert "4 steps" in FULL_SYSTEM and "4 steps" in DOM_ONLY_SYSTEM and "tool calls" not in FULL_SYSTEM.split("after 4")[1][:20]


def test_dom_only_prompt_differs_only_by_image_and_tools():
    """The DOM-only system prompt omits only the image description and the vision/verify tools."""
    from parallax.model.prompts import TOOL_DOCS
    full_only = [
        "a screenshot of the same page with those ids drawn as small yellow labels at each element's top-left corner, ",
        "; the screenshot shows what is really there. verify_target tells you whether they agree",
        f"\n- {TOOL_DOCS['click_coord']}\n- {TOOL_DOCS['verify_target']}",
        "Tools (call exactly one per turn):",
        '{\"tool\": <name>, \"args\": {...}}',
    ]
    dom_only = [
        " and its box (x,y,w,h)",
        "Tool (call exactly one per turn):",
        '{\"tool\": \"click_element\", \"args\": {\"node_id\": <int>}}',
    ]
    f, d = FULL_SYSTEM, DOM_ONLY_SYSTEM
    for x in full_only:
        assert x in f, x
        f = f.replace(x, "")
    for x in dom_only:
        assert x in d, x
        d = d.replace(x, "")
    assert " ".join(f.split()) == " ".join(d.split())
