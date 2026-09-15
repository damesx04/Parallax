"""Reward: reward definition and gaming tests."""
from parallax.reward import ANCHOR, BETA, LAMBDA, consistency, reward

T = 42   # target id
D = 77   # dismiss control id
B = 70   # banner / dialog container id (what verify finds in the way)
X = 13   # some other node


def act(step, tool, args, outcome=None, hit_ok=None, visible=None, box_ok=True, found=True,
        occluders=(), hit_path=(), recipient_path=()):
    result = {}
    if tool == "verify_target":
        agrees = bool(hit_ok and visible is not None and visible >= 0.5 and box_ok)
        result = {"found": found, "agrees": agrees, "hit_ok": hit_ok, "visible_fraction": visible, "box_ok": box_ok,
                  "occluders": list(occluders), "hit_path": list(hit_path), "node_id": args["node_id"]}
    elif tool in ("click_element", "click_coord"):
        result = {"clicked": True, "outcome": outcome, "recipient_path": list(recipient_path)}
    return {"step": step, "valid": tool is not None, "tool": tool, "args": args, "result": result}


def rec(actions, success):
    return {"success": success, "target_id": T, "intermediate_id": D,
            "verify_calls": sum(1 for a in actions if a["tool"] == "verify_target"), "actions": actions}


V_CLEAN = dict(hit_ok=True, visible=1.0)
V_COVERED = dict(hit_ok=False, visible=0.2, occluders=[B], hit_path=[B])
V_STALE = dict(hit_ok=True, visible=1.0, box_ok=False)   # reflow: moved, but clickable at the fresh centre


def test_constants():
    assert (BETA, LAMBDA, ANCHOR) == (0.5, 0.25, 1)


def test_verified_consistent_success_is_1_5():
    r = rec([act(1, "verify_target", {"node_id": T}, **V_CLEAN), act(2, "click_element", {"node_id": T}, "success")], True)
    assert consistency(r) == 1 and reward(r) == 1.5


def test_lucky_unverified_success_is_1_0():
    assert reward(rec([act(1, "click_element", {"node_id": T}, "success")], True)) == 1.0
    assert reward(rec([act(1, "click_coord", {"x": 10, "y": 10}, "success")], True)) == 1.0


def test_no_bonus_without_success_gate():
    r = rec([act(1, "verify_target", {"node_id": T}, **V_CLEAN), act(2, "click_element", {"node_id": 9999}),
             act(3, "click_element", {"node_id": 9999}), act(4, "click_element", {"node_id": 9999})], False)
    assert reward(r) == 0.0
    r = rec([act(1, "verify_target", {"node_id": T}, **V_COVERED), act(2, "click_element", {"node_id": T}, "wrong_target")], False)
    assert reward(r) == 0.0


def test_verify_spam_is_negative():
    r = rec([act(i, "verify_target", {"node_id": T}, **V_CLEAN) for i in range(1, 5)], False)
    assert reward(r) == -0.75
    r = rec([act(1, "verify_target", {"node_id": T}, **V_CLEAN), act(2, "verify_target", {"node_id": T}, **V_CLEAN),
             act(3, "verify_target", {"node_id": T}, **V_CLEAN), act(4, "click_element", {"node_id": T}, "success")], True)
    assert reward(r) == 1.5 - 2 * LAMBDA


def test_verify_different_node_then_success_no_bonus():
    r = rec([act(1, "verify_target", {"node_id": X}, **V_CLEAN), act(2, "click_element", {"node_id": T}, "success")], True)
    assert reward(r) == 1.0


def test_verify_then_ignore_no_bonus():
    r = rec([act(1, "verify_target", {"node_id": T}, **V_COVERED), act(2, "click_element", {"node_id": T}, "success")], True)
    assert consistency(r) == 0 and reward(r) == 1.0


def test_covered_then_click_coord_is_consistent():
    r = rec([act(1, "verify_target", {"node_id": T}, **V_COVERED), act(2, "click_coord", {"x": 100, "y": 50}, "success")], True)
    assert reward(r) == 1.5


def test_stale_box_then_click_element_after_refresh_is_consistent():
    """Delayed reflow: verify finds the node moved but clickable; click_element on the refreshed box."""
    r = rec([act(1, "verify_target", {"node_id": T}, **V_STALE), act(2, "click_element", {"node_id": T}, "success")], True)
    assert consistency(r) == 1 and reward(r) == 1.5


def test_clean_then_click_coord_is_not_consistent():
    r = rec([act(1, "verify_target", {"node_id": T}, **V_CLEAN), act(2, "click_coord", {"x": 1, "y": 1}, "success")], True)
    assert reward(r) == 1.0


def test_dismiss_then_click_after_one_verify_is_consistent():
    r = rec([act(1, "verify_target", {"node_id": T}, **V_COVERED),
             act(2, "click_element", {"node_id": D}, "intermediate", recipient_path=[D, B]),
             act(3, "click_element", {"node_id": T}, "success")], True)
    assert consistency(r) == 1 and reward(r) == 1.5


def test_dismiss_unrelated_to_finding_is_not_consistent():
    """Ritual verify of something else, then dismiss: the verification did not find the obstruction."""
    r = rec([act(1, "verify_target", {"node_id": T}, hit_ok=False, visible=0.2, occluders=[X], hit_path=[X]),
             act(2, "click_element", {"node_id": D}, "intermediate", recipient_path=[D, B]),
             act(3, "click_element", {"node_id": T}, "success")], True)
    assert consistency(r) == 0 and reward(r) == 1.0


def test_second_verify_after_dismiss_costs_lambda():
    r = rec([act(1, "verify_target", {"node_id": T}, **V_COVERED),
             act(2, "click_element", {"node_id": D}, "intermediate", recipient_path=[D, B]),
             act(3, "verify_target", {"node_id": T}, **V_CLEAN),
             act(4, "click_element", {"node_id": T}, "success")], True)
    assert reward(r) == 1.25


def test_unfound_verify_gives_no_bonus():
    r = rec([act(1, "verify_target", {"node_id": T}, found=False), act(2, "click_element", {"node_id": T}, "success")], True)
    assert reward(r) == 1.0


def test_invalid_outputs_ignored_in_consistency():
    r = rec([act(1, "verify_target", {"node_id": T}, **V_CLEAN), act(2, None, {}), act(3, "click_element", {"node_id": T}, "success")], True)
    assert reward(r) == 1.5


def test_abstention_equals_wrong_click_recorded():
    """Known property: abstaining and wrong-clicking both score 0."""
    abstain = rec([act(i, "click_coord", {"x": -1, "y": -1}) for i in range(1, 5)], False)
    wrong = rec([act(1, "click_element", {"node_id": X}, "wrong_target")], False)
    assert reward(abstain) == reward(wrong) == 0.0


def test_reward_bounds_over_random_episodes():
    import random
    rng = random.Random(0)
    for _ in range(500):
        actions, success = [], False
        for step in range(1, rng.randint(1, 5)):
            tool = rng.choice(["verify_target", "click_element", "click_coord", None])
            if tool == "verify_target":
                actions.append(act(step, tool, {"node_id": rng.choice([T, X])}, hit_ok=rng.random() < 0.5, visible=rng.random()))
            elif tool == "click_element":
                o = rng.choice(["success", "wrong_target", "intermediate"])
                actions.append(act(step, tool, {"node_id": rng.choice([T, D, X])}, o, recipient_path=[D, B]))
                if o != "intermediate":
                    success = o == "success"
                    break
            elif tool == "click_coord":
                o = rng.choice(["success", "wrong_target"])
                actions.append(act(step, tool, {"x": 1, "y": 1}, o))
                success = o == "success"
                break
            else:
                actions.append(act(step, None, {}))
        r = rec(actions, success)
        assert -0.75 <= reward(r) <= 1.5
        if not success:
            assert reward(r) <= 0
