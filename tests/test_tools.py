"""Tools: exactly three tools with fixed names and argument schemas."""
import pytest

from parallax import tools
from parallax.tools import InvalidAction, TOOL_NAMES, TOOLS, validate_action


def test_exactly_three_tools():
    assert TOOL_NAMES == ("click_element", "click_coord", "verify_target")
    assert set(TOOLS) == set(TOOL_NAMES) and len(TOOLS) == 3
    public = [n for n in dir(tools) if not n.startswith("_") and callable(getattr(tools, n))]
    assert set(public) <= {"validate_action", "InvalidAction"}


def test_schemas():
    assert TOOLS["click_element"] == {"node_id": int}
    assert TOOLS["click_coord"] == {"x": int, "y": int}
    assert TOOLS["verify_target"] == {"node_id": int}


def test_validate_ok_and_coerce():
    assert validate_action({"tool": "click_element", "args": {"node_id": "12"}}) == {"tool": "click_element", "args": {"node_id": 12}}
    assert validate_action({"tool": "click_coord", "args": {"x": 10.4, "y": 20}})["args"] == {"x": 10, "y": 20}


@pytest.mark.parametrize("bad", [
    {"tool": "scroll", "args": {}},
    {"tool": "wait", "args": {"ms": 1000}},
    {"tool": "click_element", "args": {}},
    {"tool": "click_element", "args": {"node_id": 1, "extra": 2}},
    {"tool": "click_coord", "args": {"x": "a", "y": 1}},
    {"tool": "click_element", "args": {"node_id": True}},
    "click_element(1)",
    None,
])
def test_validate_rejects(bad):
    with pytest.raises(InvalidAction):
        validate_action(bad)


def test_restricted_tool_set():
    with pytest.raises(InvalidAction):
        validate_action({"tool": "verify_target", "args": {"node_id": 1}}, allowed=("click_element",))
