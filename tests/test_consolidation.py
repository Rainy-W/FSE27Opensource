import tempfile
import unittest
from pathlib import Path

import torch

from humer.config import HumerConfig
from humer.data import CodeSample
from humer.runner import HumerRunner


class FakeModel:
    def __init__(self) -> None:
        self.weight = torch.tensor([50.0])

    def state_dict(self):
        return {"weight": self.weight.clone()}

    def load_state_dict(self, state):
        self.weight = state["weight"].clone()


class ConsolidationTests(unittest.TestCase):
    def test_final_checkpoint_requires_a_consolidation_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = object.__new__(HumerRunner)
            runner.config = HumerConfig(
                data_dir="data",
                output_dir=directory,
                model="codebert",
                batch_size=2,
                consolidation_max_epochs=3,
                consolidation_patience=2,
            )
            runner.output_dir = Path(directory)
            runner.train_samples = [
                CodeSample("a", 0, 1),
                CodeSample("b", 1, 2),
            ]
            runner.val_samples = [CodeSample("validation", 1, 3)]
            runner.total_optimizer_steps = 10
            runner.eval_step_schedule = []
            runner.consolidation_rows = []
            runner.selected_checkpoint = "stage_5_best.pt"
            runner.selected_validation_metrics = {"f1": 0.99, "macro_f1": 0.99}
            runner.selected_consolidation_epoch = None
            runner._create_optimizer = lambda model: object()
            runner._log = lambda message: None

            scores = iter((0.4, 0.6, 0.5))

            def train_plain(model, optimizer, samples, steps_per_epoch, epochs):
                model.weight += 1
                runner.total_optimizer_steps += steps_per_epoch * epochs

            def evaluate(model, samples):
                score = next(scores)
                return {
                    "accuracy": score,
                    "precision": score,
                    "recall": score,
                    "f1": score,
                    "macro_f1": score,
                    "mcc": score,
                }

            runner._train_plain = train_plain
            runner._evaluate = evaluate
            model = runner._run_consolidation(FakeModel())

            self.assertEqual(runner.selected_consolidation_epoch, 2)
            self.assertEqual(float(model.weight.item()), 52.0)
            self.assertTrue(runner.selected_checkpoint.endswith("consolidation_best.pt"))
            self.assertNotEqual(runner.selected_checkpoint, "stage_5_best.pt")


if __name__ == "__main__":
    unittest.main()
