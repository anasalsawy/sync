"""Task groups: one task → three R; three tasks → nine R."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from hands_paths import PATH_APPLICATION, PATH_BROWSER, PATH_SHELL, build_hands_path
from shared_resources import SharedResourceBus
from task_groups import MainRCoordinator


class TestTaskGroups(unittest.IsolatedAsyncioTestCase):
    async def test_three_tasks_nine_r_peak(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bus = SharedResourceBus()
            main = MainRCoordinator(
                bus=bus, workers_per_group=3, root=root, activation_injection=True,
            )
            specs = [
                {
                    "goal": "Browser task: navigate, click, type",
                    "path": PATH_BROWSER,
                    "host": build_hands_path(
                        PATH_BROWSER, root=root / "b", bus=bus, browser_delay_s=0.08,
                    ),
                },
                {
                    "goal": "Application task: open, type, save",
                    "path": PATH_APPLICATION,
                    "host": build_hands_path(
                        PATH_APPLICATION, root=root / "a", bus=bus, app_delay_s=0.08,
                    ),
                },
                {
                    "goal": "Shell task: list, write, verify",
                    "path": PATH_SHELL,
                    "host": build_hands_path(PATH_SHELL, root=root / "s", bus=bus),
                },
            ]
            groups = await main.start_many(specs, wait=False)
            peak = 0
            peak_g = 0
            while any(g.status in ("starting", "running") for g in groups):
                snap = main.snapshot()
                peak = max(peak, snap["active_r_workers_total"])
                peak_g = max(peak_g, snap["active_task_groups"])
                await asyncio.sleep(0.002)
            await main.wait_all()
            self.assertEqual(peak_g, 3)
            self.assertEqual(peak, 9)
            self.assertTrue(all(g.n_workers == 3 for g in groups))
            self.assertTrue(all(g.result and g.result.satisfied for g in groups))
            browser = next(g for g in groups if g.path == PATH_BROWSER)
            app = next(g for g in groups if g.path == PATH_APPLICATION)
            shell = next(g for g in groups if g.path == PATH_SHELL)
            self.assertEqual(browser.host_launch.get("host_mode"), "single_path")
            self.assertEqual(app.host_launch.get("host_mode"), "native_subagent")
            self.assertEqual(app.host_launch.get("n_subagents"), 3)
            self.assertEqual(shell.host_launch.get("host_mode"), "single_path")


if __name__ == "__main__":
    unittest.main()
