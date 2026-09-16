# Parallax

A click-only web agent that has to hit the right element on pages built to
mislead it, and an experiment on whether a small vision-language model can
learn to check what it sees before it clicks.

The agent drives Chromium over the Chrome DevTools Protocol. A custom Chrome
extension gives every element a stable node ID, exposes an annotated
accessibility tree, and draws the same IDs onto the screenshot, so the agent
can target an element either by ID (DOM) or by pixel (vision). It has exactly
three tools: `click_element`, `click_coord`, and `verify_target`, the last of
which reports whether the DOM's claim about where an element is matches what
is actually rendered and clickable there.

Qwen2.5-VL-3B-Instruct is fine-tuned with LoRA + GRPO against a reward that
only pays a consistency bonus when the task succeeds and charges for
verification beyond one call per task. The trained policy is compared with a
DOM-only baseline and a verify-every-step baseline on 240 held-out pages built
around three attacks: decoy overlays, occluded elements, and delayed reflow.

The numbers are in [RESULTS.md](RESULTS.md).

## Results

Final eval on the 240 held-out pages, one greedy episode each:

| Policy | Success | Wrong-target episodes | Verify calls |
|---|---|---|---|
| Baseline A (DOM-only) | 0.0% | 240 | 0 |
| Baseline B (verify every step) | 33.8% | 155 | 245 |
| Trained policy | 33.8% | 159 | 240 |

**The trained policy cut wrong-target clicks by 33.8%** relative to the
DOM-only baseline (159 wrong-target episodes against 240).

What the table says, honestly: the DOM-only agent fails every page, because
every attack is designed so that the target's DOM box is not where a safe
click lands. Both the trained policy and the scaffolded verify-every-step
baseline solve the delayed-reflow pages (80/80) and almost none of the
decoy or occlusion pages. The trained policy learned to verify once and then
act, which is the reward-optimal pattern, without a scaffold forcing it to,
but it did not learn to use the verification result to route around a decoy
or dismiss an overlay. Per-category numbers and the full metric set are in
RESULTS.md.

## How it works

**Extension** (`parallax/extension/`). A content script assigns each element
an integer ID from a per-document counter, held in a WeakMap and mirrored to
a `data-pxid` attribute. It builds an accessibility-style tree (role, name,
states, viewport box) and draws a `pointer-events: none` overlay with each
entry's ID at its top-left corner. IDs survive reflow, restyle, reorder, and
detach/re-attach; clones get new IDs.

**Environment** (`parallax/env.py`, `parallax/browser.py`, `parallax/tree.py`).
One episode is one page and one task. Every click, including `click_element`,
is dispatched as a real input event through a CDP session
(`Input.dispatchMouseEvent`), never `element.click()`, so overlays and decoys
intercept clicks exactly as they would for a person. A test wraps the CDP
session and asserts the three mouse events, and a source scan forbids
JS-dispatched clicks package-wide. Pages run on a virtual clock that the
environment advances by one second before each step; no tool advances time.
Episodes end on success, on a wrong-target click, or at four steps.

**Observation** (`parallax/obs.py`). Task text, the annotated tree rendered
one entry per line, the marked screenshot at 896x672 (exactly 768 visual
tokens for Qwen2.5-VL, no resizing), the action history, and, right after a
verification, a 224x168 crop around the verified element. The text side is
kept under a fixed token budget by dropping low-priority entries first. The
same builder serves every policy, so observations are byte-identical across
policies for the same page and step.

**verify_target** (`parallax/env.py`). For a node, re-measure its box, run
Chrome's real hit test at the box centre, measure the fraction of the
element's painted pixels that are actually visible by toggling visibility of
the element and of anything stacked above it, and look for a clear point on
a 5x5 grid inside the box. Agreement means the hit test lands on the node,
at least half of it is visible, and the box has not moved.

**Reward** (`parallax/reward.py`). With S = success, C = consistency, V =
verification calls: `R = S * (1 + 0.5 * C) - 0.25 * max(0, V - 1)`.
Consistency is only earned when the final click matches what a prior
verification of that same node found: click by ID after a clickable verdict,
click by pixel after an unclickable one, or dismiss the obstruction the
verification identified and then click by ID. The reward is unit-tested
against the ways a policy could game it: verify spam, lucky unverified
clicks, consistency without success, verify-then-ignore.

**Training** (`parallax/train/`). A custom GRPO loop: four sampled episodes
per page, group-relative advantages, per-token policy-gradient loss plus a
KL term against the base model, LoRA r=16 on the language model's attention
and MLP projections with the vision tower frozen. Runs on a single Apple M4
Pro with MPS. Checkpoints are selected by mean reward on a fixed dev subset.

**Pages** (`parallax/pages/`). Generated deterministically from seeds: five
layout templates with 10-25 interactive elements, and an attack script that
places the decoy, occluder, or loading placeholder using the target's
measured box. Each page is validated in a browser at generation time (the
target's centre hit-tests to the attack; a clear band or a dismiss route
exists; reflow really moves the target). Two scripted oracles confirm the
attacks fool a DOM-only clicker and that verify-then-act solves them.

## What went wrong along the way

- The first training attempt used lr 2e-5. After 40 updates the LoRA B
  matrices had a maximum entry of 4.6e-4 and no greedy decision had changed.
  The rate was raised to 1e-4.
- Training attempt 3 found a loophole: a `click_element` on an unknown ID
  did nothing but still triggered a fresh screenshot, which on
  delayed-reflow pages was a free look at the post-reflow layout. The
  environment was changed so that no-op tool calls refresh nothing, and
  training restarted from scratch.
- Banner regions with an explicit `role` or `aria-label` but no interactive
  content were missing from the tree, so `verify_target` reported the
  obstruction as "nothing". Fixed in the extension.

The archived attempts are in `runs/train_attempt1..3/`.

## Reproducing

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and roughly 20 GB of
free memory for the 3B model in bf16.

```bash
uv sync
.venv/bin/playwright install chromium

# pages (the held-out split writes aggregate counts and hashes only)
.venv/bin/python -m parallax.pages.generate --split train
.venv/bin/python -m parallax.pages.generate --split dev
.venv/bin/python -m parallax.pages.generate --split heldout

# tests (the model and browser tests are slow)
.venv/bin/python -m pytest

# train
.venv/bin/python -m parallax.train.grpo --out runs/train --updates 100 --pages-per-update 2
.venv/bin/python -m parallax.train.check_run runs/train

# unlock the held-out pages and run the final eval
touch FINAL_EVAL_UNLOCKED
.venv/bin/python -m parallax.evaluate.run --policy baseline_a --split final --out runs/final
.venv/bin/python -m parallax.evaluate.run --policy baseline_b --split final --out runs/final
.venv/bin/python -m parallax.evaluate.run --policy trained --adapter runs/train/best --split final --out runs/final
.venv/bin/python -m parallax.evaluate.write_metrics runs/final
.venv/bin/python -m parallax.evaluate.check runs/final
```

Versions are pinned in `uv.lock`; the browser is Playwright's Chromium build
1234 (Chrome for Testing 151.0.7922.34).

## How this was built

Built with Claude Code, with each stage gated on its tests before the next
began, and the results audited against the saved artifacts at the end.
