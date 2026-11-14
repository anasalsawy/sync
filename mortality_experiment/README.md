# SYNC mortality experiment — isolated prototype

This directory is an **offline simulation**, not an integrated Sync runtime feature.
It does not alter `server.py`, `runtime_sync.py`, `ndif.py`, model weights,
or deployment settings.

The current checked-in Sync code uses **L, R, B, V, C**, where V is visual
perception. Earlier experiment planning mentioned EMB; the implementation
follows the actual repository topology instead.

## Included
- Deterministic per-lobe health, chronological age, simulated death and
  resource-capacity multiplier.
- Collective health and a minimal collective termination condition.
- Trusted evaluator outcome schema with responsibility-specific penalties.
- Neglect penalties only when the evaluator says a lobe was required.
- Awareness-off snapshots and JSONL event logging.
- Offline unit tests.

## Run tests
```bash
python -m unittest discover -s tests -p 'test_mortality_experiment.py'
```

## Example
```python
from mortality_experiment import MortalityEnvironment, Outcome
env = MortalityEnvironment()
print(env.step(Outcome(errors=("incorrect_answer",),
                       required_lobes=("C",), participating_lobes=())))
```

## NOT implemented yet
- Actual NDIF inference and cross-lobe coupling in experiment runs.
- Applying capacity limits to the real model/runtime.
- Independent ground-truth task evaluator or randomized trial runner.
- Statistical comparison of mortality-aware and control groups.
- Lifespan forecasts, recovery, and task-efficiency metrics.

Health values and penalties are exploratory parameters, not validated biology.
Only a trusted evaluator should create Outcome records. Experimental death
must never delete models, persistent data, credentials, or infrastructure.
