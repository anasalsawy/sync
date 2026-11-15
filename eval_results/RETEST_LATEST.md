# Retest on GitHub `main` @ `beef896` (2026-11-14)

Pulled from `origin/main` (5 commits ahead of prior local). Model: `Qwen/Qwen3-8B`.
Harness: solo = L-only generate; connected = L+R+B+V+C; `--repeats 3`, `--max-tokens 64`.

## Efficacy

| | Prior baseline | This retest |
|---|---|---|
| solo pass | 4/8 | **4/8** |
| connected pass | **7/8** | **0/8** |
| connected helps | 7/8 | **0/8** |

Connected often emits unclosed `<think>…` (cleaned to empty) or ignores peer facts
even when it speaks. Prefill-only channel writes in current `ndif.generate_sync`
are much weaker than the earlier continuous mind-read path.

## Latency (24 samples/condition)

| | mean | p50 | p95 |
|---|---|---|---|
| solo | 5.63s | 6.05s | 6.99s |
| connected | 11.56s | 11.39s | 13.20s |
| overhead (conn−solo) | **+5.93s** | | |

Full JSON: `eval_results/retest_latest.json`
