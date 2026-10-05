import unittest

from humer.review import adaptive_history_ratio, stage_slot_schedule


class ReviewTests(unittest.TestCase):
    def test_stage_schedule_matches_continuous_budgets(self) -> None:
        schedule, budgets = stage_slot_schedule(
            batch_size=32, review_steps=17, history_ratio=0.413, error_ratio=0.087
        )
        self.assertEqual(sum(current for current, _, _ in schedule), budgets["current_budget"])
        self.assertEqual(
            sum(general + error for _, general, error in schedule), budgets["history_budget"]
        )
        self.assertEqual(sum(error for _, _, error in schedule), budgets["error_budget"])
        self.assertEqual(len(schedule), 17)

    def test_adaptive_ratio_is_not_clipped(self) -> None:
        ratio = adaptive_history_ratio(historical_loss=9.0, new_loss=1.0, forgetting=0.0)
        self.assertGreater(ratio, 0.75)


if __name__ == "__main__":
    unittest.main()
