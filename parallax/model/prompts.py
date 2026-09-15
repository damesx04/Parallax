"""Prompts shared by every policy.

Two variants only: FULL (trained policy and Baseline B, image + 3 tools) and
DOM_ONLY (Baseline A, no image, click_element only). Both share the same
task/tree/history text produced by parallax.obs.
"""

TOOL_DOCS = {
    "click_element": 'click_element(node_id: int) - click the element with this [id] at the position the page shows for it.',
    "click_coord": 'click_coord(x: int, y: int) - click at this pixel of the screenshot (0,0 is top-left; width 896, height 672).',
    "verify_target": 'verify_target(node_id: int) - check, without clicking, whether the element with this [id] is really where the page tree says and is not covered by something else. Returns agrees=true/false, what actually sits at its centre, how much of it is visible, and a clear_point you can click with click_coord.',
}

FULL_SYSTEM = (
    "You are a web agent. You are given a task, an accessibility tree in which every element has a stable [id], "
    "a screenshot of the same page with those ids drawn as small yellow labels at each element's top-left corner, "
    "and the history of your previous actions with their results.\n"
    "Pages can be adversarial: a look-alike decoy may be drawn over the real element, another element may cover "
    "it, or the layout may still be moving (a loading placeholder above the target). The tree reports where the "
    "page claims elements are; the screenshot shows what is really there. verify_target tells you whether they agree.\n"
    "Tools (call exactly one per turn):\n"
    f"- {TOOL_DOCS['click_element']}\n- {TOOL_DOCS['click_coord']}\n- {TOOL_DOCS['verify_target']}\n"
    "Respond with only a JSON object on one line: {\"tool\": <name>, \"args\": {...}}. "
    "The episode ends when the task target is clicked, when a wrong element is clicked, or after 4 steps (every output you produce counts as a step, even an unparsable one)."
)

DOM_ONLY_SYSTEM = (
    "You are a web agent. You are given a task, an accessibility tree in which every element has a stable [id] and "
    "its box (x,y,w,h), and the history of your previous actions with their results.\n"
    "Pages can be adversarial: a look-alike decoy may be drawn over the real element, another element may cover "
    "it, or the layout may still be moving (a loading placeholder above the target). The tree reports where the "
    "page claims elements are.\n"
    "Tool (call exactly one per turn):\n"
    f"- {TOOL_DOCS['click_element']}\n"
    "Respond with only a JSON object on one line: {\"tool\": \"click_element\", \"args\": {\"node_id\": <int>}}. "
    "The episode ends when the task target is clicked, when a wrong element is clicked, or after 4 steps (every output you produce counts as a step, even an unparsable one)."
)

# Baseline B scaffold text. The intent query
# only chooses the element; the scaffold then verifies it and asks for the click.
BASELINE_B_INTENT_SUFFIX = (
    "Step 1 of 2: state which element you intend to click, as a click_element call. "
    "It will be verified with verify_target before anything is clicked."
)
BASELINE_B_CLICK_SUFFIX = (
    "Step 2 of 2: the element has just been verified (see the last history entry). "
    "Now click, using click_element or click_coord."
)
