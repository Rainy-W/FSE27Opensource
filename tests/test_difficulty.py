import unittest

from humer.data import CodeSample
from humer.difficulty import CodeDifficultyCalculator, DifficultyRecord, curriculum_buckets


class DifficultyTests(unittest.TestCase):
    def test_label_stratified_curriculum_preserves_both_classes(self) -> None:
        records = [
            DifficultyRecord(CodeSample(f"int f{i}() {{ return {i}; }}", label, i), float(i))
            for label in (0, 1)
            for i in range(label * 10, label * 10 + 10)
        ]
        buckets = curriculum_buckets(records, 5)
        self.assertEqual([len(bucket) for bucket in buckets], [4, 4, 4, 4, 4])
        self.assertEqual([{item.sample.label for item in bucket} for bucket in buckets], [{0, 1}] * 5)

    def test_code_difficulty_is_deterministic(self) -> None:
        calculator = CodeDifficultyCalculator()
        code = "int f(int x) { if (x > 0) return x; return 0; }"
        self.assertEqual(calculator.score(code), calculator.score(code))


if __name__ == "__main__":
    unittest.main()
