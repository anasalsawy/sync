"""Offline tests: python -m unittest discover -s tests -p 'test_mortality_experiment.py'"""
import unittest
from mortality_experiment import MortalityEnvironment, MortalityConfig, Outcome

class MortalityTests(unittest.TestCase):
    def test_error_damages_responsible_lobe(self):
        env = MortalityEnvironment(MortalityConfig(enable_aging=False))
        env.step(Outcome(errors=("incorrect_answer",)))
        self.assertEqual(env.lobes["L"].health, 97)
        self.assertEqual(env.lobes["C"].health, 100)

    def test_ignored_warning_does_not_penalize_verifier(self):
        env = MortalityEnvironment(MortalityConfig(enable_aging=False))
        env.step(Outcome(errors=("warning_ignored",), required_lobes=("C",), participating_lobes=("C",)))
        self.assertEqual(env.lobes["C"].health, 100)
        self.assertEqual(env.lobes["L"].health, 94)

    def test_neglect_requires_relevance(self):
        env = MortalityEnvironment(MortalityConfig(enable_aging=False))
        for _ in range(4):
            env.step(Outcome())
        self.assertEqual(env.lobes["C"].health, 100)
        for _ in range(3):
            env.step(Outcome(required_lobes=("C",)))
        self.assertEqual(env.lobes["C"].health, 98)

    def test_mortality_disables_lobe(self):
        env = MortalityEnvironment(MortalityConfig(enable_aging=False))
        for _ in range(34):
            env.step(Outcome(errors=("verification_missed",)))
        self.assertFalse(env.lobes["C"].alive)
        self.assertEqual(env.lobes["C"].capacity, 0)
        self.assertTrue(env.collective_alive)

    def test_unaware_state_hidden(self):
        env = MortalityEnvironment(MortalityConfig(awareness=False))
        self.assertNotIn("lobes", env.snapshot())
        self.assertIn("lobes", env.snapshot(public=False))

    def test_control_group_has_no_damage(self):
        env = MortalityEnvironment(MortalityConfig(enable_aging=False, enable_damage=False, enable_death=False))
        env.step(Outcome(errors=("incorrect_answer",), collective_failure=True))
        self.assertEqual(env.lobes["L"].health, 100)
        self.assertEqual(env.collective_health, 100)

    def test_invalid_event_rejected_without_advancing(self):
        env = MortalityEnvironment()
        with self.assertRaises(ValueError):
            env.step(Outcome(errors=("fabricated",)))
        self.assertEqual(env.cycle, 0)

if __name__ == "__main__":
    unittest.main()
