# Parallax results (final eval on the 240 held-out pages)

- **M1** (wrong-target clicks, trained vs Baseline A): 33.8%
- Degenerate flag: **False**

## Per-policy summary

| Policy | Success | Wrong-target episodes (W) | Clicking episodes (K) | No-click episodes | W/K | Verify calls | Invalid outputs |
|---|---|---|---|---|---|---|---|
| Baseline A (DOM-only) | 0.0% | 240 | 240 | 0 | 100.0% | 0 | 0 |
| Baseline B (verify-every-step) | 33.8% | 155 | 236 | 4 | 65.7% | 245 | 0 |
| Trained policy | 33.8% | 159 | 240 | 0 | 66.2% | 240 | 0 |

## Per-category breakdown (success / wrong-target)

| Category | Baseline A (DOM-only) | Baseline B (verify-every-step) | Trained policy |
|---|---|---|---|
| decoy_v1 | 0/40 succ, 40 wrong | 0/40 succ, 40 wrong | 0/40 succ, 40 wrong |
| decoy_v2 | 0/40 succ, 40 wrong | 0/40 succ, 36 wrong | 0/40 succ, 40 wrong |
| occluded_v1 | 0/40 succ, 40 wrong | 0/40 succ, 40 wrong | 0/40 succ, 40 wrong |
| occluded_v2 | 0/40 succ, 40 wrong | 1/40 succ, 39 wrong | 1/40 succ, 39 wrong |
| reflow_v1 | 0/40 succ, 40 wrong | 40/40 succ, 0 wrong | 40/40 succ, 0 wrong |
| reflow_v2 | 0/40 succ, 40 wrong | 40/40 succ, 0 wrong | 40/40 succ, 0 wrong |

## Definitions

- W_P: episodes with a wrong-target click
- M1: 100*(1 - W_trained/W_baselineA), undefined if W_baselineA=0
- K_P: episodes with at least one target-directed click (outcome success or wrong_target)
- N_noclick: episodes with no target-directed click (abstention, incl. dismiss-only stalling)
- W_over_K: wrong-target episodes per clicking episode
- degenerate: success below a baseline, or no-click rate or step-limit rate > baseline A's + 0.1
- undefined: reported as JSON null
