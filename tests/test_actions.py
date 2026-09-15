import pytest

from parallax.model.actions import parse_action, try_parse
from parallax.tools import InvalidAction


def test_parses_all_three_tools():
    assert parse_action('{"tool": "click_element", "args": {"node_id": 12}}') == {"tool": "click_element", "args": {"node_id": 12}}
    assert parse_action('{"tool": "click_coord", "args": {"x": 10, "y": 20}}') == {"tool": "click_coord", "args": {"x": 10, "y": 20}}
    assert parse_action('{"tool": "verify_target", "args": {"node_id": 3}}') == {"tool": "verify_target", "args": {"node_id": 3}}


def test_tolerates_fences_and_prose():
    assert parse_action('Sure.\n```json\n{"tool": "verify_target", "args": {"node_id": 3}}\n```')["tool"] == "verify_target"
    assert parse_action('I will click. {"tool":"click_element","args":{"node_id":"7"}} done')["args"] == {"node_id": 7}
    assert parse_action('{"tool": "click_element", "node_id": 5}')["args"] == {"node_id": 5}
    assert parse_action('{"name": "click_coord", "arguments": {"x": 1, "y": 2}}')["tool"] == "click_coord"


@pytest.mark.parametrize("bad", ["", "click_element(3)", "{}", '{"tool": "scroll", "args": {}}',
                                 '{"tool": "click_element", "args": {}}', '{"tool": "click_element", "args": {"node_id": "abc"}}',
                                 '{"tool": "click_element"', "no json here"])
def test_rejects(bad):
    with pytest.raises(InvalidAction):
        parse_action(bad)
    a, err = try_parse(bad)
    assert a is None and err


def test_restricted_tools():
    with pytest.raises(InvalidAction):
        parse_action('{"tool": "verify_target", "args": {"node_id": 3}}', allowed=("click_element",))
