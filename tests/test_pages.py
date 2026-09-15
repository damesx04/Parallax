"""Pages: deterministic generation, counts, seeds, hashes, summary invariants."""
import json


from parallax.pages import generate as gen
from parallax.pages.loader import DEV_DIR, HOLDOUT_SPLIT, HOLDOUT_SUMMARY, TRAIN_DIR, sha256_file

TOKEN = "held" + "out"


def test_render_is_deterministic():
    a, ia = gen.render_page("checkout", 123, "decoy", 1, 0)
    b, ib = gen.render_page("checkout", 123, "decoy", 1, 0)
    assert a == b and ia == ib
    c, _ = gen.render_page("checkout", 124, "decoy", 1, 0)
    assert c != a


def test_generated_split_bytes_match_regeneration():
    """Same seed -> identical bytes (train/dev only; held-out bytes are never read here)."""
    for d in (TRAIN_DIR, DEV_DIR):
        summary = json.loads((d / "summary.json").read_text())
        for pid in summary["hashes"]:
            m = json.loads((d / pid / "manifest.json").read_text())
            doc, _ = gen.render_page(m["template"], m["seed"], m["category"], m["variant"], m["attempt"])
            assert (d / pid / "page.html").read_text() == doc
            assert sha256_file(d / pid / "page.html") == summary["hashes"][pid]


def test_split_plans_and_seed_ranges_disjoint():
    tr, dv, ho = (set(gen.SPLITS[s]["seeds"]) for s in ("train", "dev", HOLDOUT_SPLIT))
    assert not (tr & ho) and not (dv & ho) and not (tr & dv)
    assert len(ho) == 240 and min(ho) == 100000 and max(ho) == 100239
    assert gen.SPLITS[HOLDOUT_SPLIT]["cycle"] == gen.VARIANTS  # no clean pages in the eval set


def test_train_and_dev_summaries():
    tr = json.loads((TRAIN_DIR / "summary.json").read_text())
    dv = json.loads((DEV_DIR / "summary.json").read_text())
    assert tr["count"] == 800 and tr["invariants_passed"] == 800
    assert dv["count"] == 160 and dv["invariants_passed"] == 160
    for k in ("decoy_v1", "decoy_v2", "occluded_v1", "occluded_v2", "reflow_v1", "reflow_v2"):
        assert tr["counts"][k] == 100 and dv["counts"][k] == 20
    assert tr["counts"]["clean_v0"] == 200 and dv["counts"]["clean_v0"] == 40
    assert len(set(tr["hashes"].values()) | set(dv["hashes"].values())) == 960  # all distinct


def test_holdout_summary_counts_seeds_hashes_and_validity():
    s = json.loads(HOLDOUT_SUMMARY.read_text())
    assert s["split"] == HOLDOUT_SPLIT and s["count"] == 240 and s["invariants_passed"] == 240
    assert s["seeds"] == [100000, 100239]
    for k in ("decoy_v1", "decoy_v2", "occluded_v1", "occluded_v2", "reflow_v1", "reflow_v2"):
        assert s["counts"][k] == 40
    assert "clean_v0" not in s["counts"]
    assert len(s["hashes"]) == 240 and len(set(s["hashes"].values())) == 240
    tr = json.loads((TRAIN_DIR / "summary.json").read_text())
    dv = json.loads((DEV_DIR / "summary.json").read_text())
    assert not (set(s["hashes"].values()) & (set(tr["hashes"].values()) | set(dv["hashes"].values())))
    o = s["oracles"]
    assert o["attacked"] == 240 and o["dom_only_wrong_rate"] >= 0.95 and o["verify_then_act_success_rate"] >= 0.95
    assert set(s) <= {"split", "count", "counts", "seeds", "hashes", "invariants_passed", "mean_attempts", "oracles"}


def test_manifest_fields_and_task_text():
    for pid in ("t0000", "t0001", "t0003"):
        m = json.loads((TRAIN_DIR / pid / "manifest.json").read_text())
        assert {"page_id", "split", "category", "variant", "seed", "template", "task", "target_selector",
                "allowed_intermediate_selector"} <= set(m)
        assert m["task"].startswith('Click the "')
        if m["variant"] == 2 and m["category"] != "reflow":
            assert m["allowed_intermediate_selector"] == "#px-dismiss"
        else:
            assert m["allowed_intermediate_selector"] is None


def test_holdout_directory_is_gitignored_and_untracked():
    import subprocess
    from parallax.pages.loader import HOLDOUT_DIR, ROOT
    rel = str(HOLDOUT_DIR.relative_to(ROOT))
    r = subprocess.run(["git", "check-ignore", "-q", rel + "/h0000/page.html"], cwd=ROOT)
    assert r.returncode == 0, "held-out directory must be gitignored"
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert tracked == ""   # meaningful once the repo has commits
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True).returncode == 0
    assert HOLDOUT_SUMMARY.exists()
