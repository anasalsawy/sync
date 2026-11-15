# Sync efficacy baseline (2026-11-12)

Models: channel=`Qwen/Qwen3-8B`, speech=`Qwen/Qwen3-8B` (27B OOM'd locally on dual load).

Source: `eval_results/baseline_full.json`

## Verdict

**Connected group clearly helps on grounding aspects** (5/6 peer-needed items).
Plain chat is not hurt. One regression: idle-peer hallucination.

| Aspect | Solo | Connected | Helps |
|---|---|---|---|
| memory_grounding | fail | pass (Room 7 path) | yes |
| environment_awareness | fail | pass (Saved) | yes |
| vision_grounding | fail | pass (Refund) | yes |
| multi_peer_synthesis | fail | pass (Zurich 10:00) | yes |
| constraint_compliance | pass | pass | yes |
| hallucination_resistance | pass | **fail** (invented 0.00%) | **no** |
| conversational_quality | pass | pass | yes |

Overall: solo 4/8, connected 7/8, connected_helps 7/8 (incl. plain chat non-regression).

## Fine-tune

`eval_finetune.py` implements NDIF remote LoRA (session + last-MLP adapter + retest).
On nnsight `0.8.2` + current NDIF, **|WB| / |bias| never move** (optimizer no-op;
confirmed with `smoke_remote_train.py`). Harness is ready; learning retest blocked
until NDIF remote training actually updates parameters.
