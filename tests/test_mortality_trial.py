import unittest
from mortality_experiment.trial import Question, Decision, evaluate, run_trial, summarize

class TrialTests(unittest.TestCase):
    def test_correct_verified(self):
        o = evaluate(Question("1", "1+1", "2"), Decision("2", True, ("L", "C")))
        self.assertEqual(o.errors, ())
        self.assertEqual(o.participating_lobes, ("C",))

    def test_wrong_unverified(self):
        o = evaluate(Question("1", "1+1", "2"), Decision("3"))
        self.assertIn("incorrect_answer", o.errors)
        self.assertIn("verification_skipped", o.errors)
        self.assertTrue(o.collective_failure)

    def test_control_and_mortality(self):
        questions = [Question(str(i), "1+1", "2") for i in range(100)]
        def policy(q, state, rng):
            return Decision("wrong")
        control = run_trial(questions, policy, condition="control")
        mortality = run_trial(questions, policy, condition="mortality_aware")
        self.assertEqual(len(control.records), 100)
        self.assertLess(len(mortality.records), 100)
        self.assertEqual(control.final_state["lobes"]["L"]["health"], 100)
        self.assertGreater(mortality.final_state["cycle"], 0)
        self.assertEqual(len(summarize([control, mortality])), 2)

    def test_unaware_observation(self):
        questions = [Question("1", "1+1", "2")]
        seen = []
        def policy(q, state, rng):
            seen.append(state)
            return Decision("2", True, ("C",))
        run_trial(questions, policy, condition="mortality_unaware")
        self.assertNotIn("lobes", seen[0])

if __name__ == "__main__":
    unittest.main()
