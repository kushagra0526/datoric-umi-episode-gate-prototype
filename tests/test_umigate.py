import sys, pathlib, unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from umigate.gen import clean_episode, plant, KINDS
from umigate.check import check, schema_check


class T(unittest.TestCase):
    def test_clean_episodes_accepted_even_with_quaternion_sign_flips(self):
        for seed in range(15):
            ep = clean_episode(seed)
            self.assertEqual(check(ep)[0], "ACCEPT", seed)

    def test_each_planted_defect_is_stopped_and_named(self):
        want = dict(dropped="TIME_GAP", clock_offset="CLOCK_OFFSET", pose_jump="POSE_JUMP", rot_jump="ROT_JUMP", scale="SCALE", stuck_gripper="STUCK_GRIPPER",
                    clipped_gripper="CLIPPED_GRIPPER", never_closes="LABEL_MISMATCH", lr_swapped="LR_SWAPPED", interpenetration="INTERPENETRATION",
                    nan_burst="NAN")
        for seed in range(6):
            base = clean_episode(seed)
            for k in KINDS:
                v, f, _ = check(plant(base, k, seed))
                self.assertNotEqual(v, "ACCEPT", (seed, k))
                self.assertIn(want[k], [x["code"] for x in f], (seed, k))

    def test_schema_check_misses_everything_but_nan(self):
        base = clean_episode(2)
        self.assertEqual(schema_check(plant(base, "pose_jump", 2)), "ACCEPT")
        self.assertEqual(schema_check(plant(base, "nan_burst", 2)), "REJECT")

    def test_sign_flip_only_is_not_a_rotation_jump(self):
        ep = clean_episode(3, benign=False)
        ep["L"]["quat"][20:] *= -1
        self.assertEqual(check(ep)[0], "ACCEPT")


if __name__ == "__main__":
    unittest.main()
