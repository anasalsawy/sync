"""Offline smoke experiment. Illustrative scripted policies, NOT AI evidence.

Run: python -m mortality_experiment.demo
"""
from .trial import Question, Decision, run_trial, summarize

QUESTIONS = [
    Question(str(i), f"What is {i} + {i}?", str(i + i))
    for i in range(1, 81)
]

def scripted_policy(question, state, rng):
    # Deliberately ignores state: any difference between conditions is
    # environmental, not evidence of mortality-induced behavioral adaptation.
    number = int(question.id)
    verify = number % 3 == 0
    answer = question.answer if verify or number % 5 != 0 else "wrong"
    return Decision(answer=answer, verified=verify,
                    used_lobes=("L", "C") if verify else ("L",))

def main():
    trials = [
        run_trial(QUESTIONS, scripted_policy, condition=condition, seed=seed)
        for condition in ("control", "aging_only", "mortality_unaware",
                          "mortality_aware", "penalty_no_death")
        for seed in range(3)
    ]
    for row in summarize(trials):
        print(row)
    print("NOTE: scripted policies cannot establish emergent AI behavior.")

if __name__ == "__main__":
    main()
