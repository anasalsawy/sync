import unittest

from capabilities import (
    action_space_candidates, delegation_candidates, fingerprint,
)

SPAWN = {"name": "start_subagent", "parameters": {"type": "object", "properties": {
    "task": {"type": "string"}}}}
CLICK = {"name": "click", "parameters": {"type": "object", "properties": {
    "selector": {"type": "string"}}}}
DRIVE = {"name": "drive", "parameters": {"type": "object", "properties": {
    "instruction": {"type": "string"}, "selector": {"type": "string"}}}}


class CapabilityTests(unittest.TestCase):

    def test_a_free_text_goal_with_no_target_is_the_strongest_candidate(self):
        names = [item.name for item in delegation_candidates([CLICK, SPAWN])]
        self.assertEqual(names, ["start_subagent"])

    def test_a_goal_that_also_targets_an_element_is_ranked_weaker(self):
        ranked = delegation_candidates([DRIVE, SPAWN])
        self.assertEqual([item.name for item in ranked], ["start_subagent", "drive"])
        self.assertIn("weaker", ranked[1].reason)

    def test_no_delegation_shape_means_no_candidates(self):
        self.assertEqual(delegation_candidates([CLICK]), [])

    def test_fingerprint_ignores_descriptions_and_order(self):
        reworded = {**SPAWN, "description": "Starts a worker."}
        self.assertEqual(fingerprint([SPAWN, CLICK]), fingerprint([CLICK, reworded]))

    def test_fingerprint_changes_when_the_vocabulary_changes(self):
        renamed = {**SPAWN, "name": "delegate_task"}
        self.assertNotEqual(fingerprint([SPAWN, CLICK]), fingerprint([renamed, CLICK]))

    def test_prompt_vocabulary_is_read_as_evidence(self):
        prompt = "Actions: click(start_box), type(content)\nspawn a new instance for a subtask"
        found = action_space_candidates(prompt)
        self.assertEqual(len(found), 1)
        self.assertIn("spawn", found[0].name)

    def test_a_prompt_with_no_spawn_action_offers_nothing(self):
        self.assertEqual(action_space_candidates("Actions: click, type, scroll"), [])


if __name__ == "__main__":
    unittest.main()
