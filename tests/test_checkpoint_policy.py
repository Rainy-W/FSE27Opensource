import unittest

from humer.runner import is_better_f1


class CheckpointPolicyTests(unittest.TestCase):
    def test_checkpoint_selection_uses_f1_then_macro_f1(self) -> None:
        best = {"f1": 0.7, "macro_f1": 0.6}
        self.assertTrue(is_better_f1({"f1": 0.71, "macro_f1": 0.5}, best))
        self.assertTrue(is_better_f1({"f1": 0.7, "macro_f1": 0.61}, best))
        self.assertFalse(is_better_f1({"f1": 0.69, "macro_f1": 0.9}, best))


if __name__ == "__main__":
    unittest.main()
