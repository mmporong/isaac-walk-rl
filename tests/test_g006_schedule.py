import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from isaac_walk_g006.mdp.curriculums import push_schedule_for_step


class PushScheduleTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(push_schedule_for_step(0), (0, 0.10, 0.25))
        self.assertEqual(push_schedule_for_step(11_999), (0, 0.10, 0.25))
        self.assertEqual(push_schedule_for_step(12_000), (1, 0.25, 0.50))
        self.assertEqual(push_schedule_for_step(23_999), (1, 0.25, 0.50))
        self.assertEqual(push_schedule_for_step(24_000), (2, 0.50, 1.00))


if __name__ == "__main__":
    unittest.main()
