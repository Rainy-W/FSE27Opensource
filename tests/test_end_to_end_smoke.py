import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from humer.config import HumerConfig
from humer.data import CodeSample, HuggingFaceCodeDataset
from humer.runner import HumerRunner


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=512, padding=False, return_attention_mask=True):
        del truncation, padding
        values = [1 + (ord(character) % 15) for character in text[:max_length]] or [1]
        return {"input_ids": values, "attention_mask": [1] * len(values)}


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4, padding_idx=0)
        self.classifier = nn.Linear(4, 2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (embedded * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        logits = self.classifier(pooled)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result


class TinyFactory:
    @staticmethod
    def create():
        return TinyClassifier()


class EndToEndSmokeTests(unittest.TestCase):
    def test_complete_release_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            config = HumerConfig(
                data_dir="unused",
                output_dir=str(output_dir),
                model="codebert",
                batch_size=2,
                eval_batch_size=2,
                max_length=32,
                learning_rate=1e-3,
                num_buckets=5,
                consolidation_max_epochs=2,
                consolidation_patience=1,
                strict_determinism=False,
            )
            config.validate()
            runner = HumerRunner(config)

            train_buckets = []
            train_samples = []
            validation_buckets = []
            validation_samples = []
            for bucket_id in range(5):
                train_bucket = [
                    CodeSample(f"train zero {bucket_id}", 0, bucket_id * 2),
                    CodeSample(f"train one {bucket_id}", 1, bucket_id * 2 + 1),
                ]
                validation_bucket = [
                    CodeSample(f"validation zero {bucket_id}", 0, 100 + bucket_id * 2),
                    CodeSample(f"validation one {bucket_id}", 1, 101 + bucket_id * 2),
                ]
                train_buckets.append(train_bucket)
                validation_buckets.append(validation_bucket)
                train_samples.extend(train_bucket)
                validation_samples.extend(validation_bucket)
            test_samples = [
                CodeSample("test zero", 0, 200),
                CodeSample("test one", 1, 201),
            ]
            tokenizer = TinyTokenizer()

            def build_data_and_model():
                runner.train_samples = train_samples
                runner.val_samples = validation_samples
                runner.test_samples = test_samples
                runner.train_dataset = HuggingFaceCodeDataset(train_samples, tokenizer, 32)
                runner.val_dataset = HuggingFaceCodeDataset(validation_samples, tokenizer, 32)
                runner.test_dataset = HuggingFaceCodeDataset(test_samples, tokenizer, 32)
                runner.train_buckets = train_buckets
                runner.validation_buckets = validation_buckets
                runner.model_factory = TinyFactory()
                runner.difficulty_info = {"type": "code", "cache_status": "smoke"}

            runner._build_data_and_model = build_data_and_model
            original_evaluate = runner._evaluate
            test_evaluation_calls = 0

            def tracked_evaluate(model, samples, include_curves=False):
                nonlocal test_evaluation_calls
                if hasattr(runner, "test_samples") and samples is runner.test_samples:
                    test_evaluation_calls += 1
                return original_evaluate(model, samples, include_curves)

            runner._evaluate = tracked_evaluate
            metrics = runner.run()

            self.assertEqual(runner.selected_consolidation_epoch, 1)
            self.assertTrue((output_dir / "consolidation_best.pt").is_file())
            self.assertTrue((output_dir / "final_test_metrics.json").is_file())
            self.assertEqual(len(runner.stage_rows), 5)
            self.assertEqual(test_evaluation_calls, 1)
            for row in runner.stage_rows[1:]:
                self.assertGreaterEqual(row["target_r_s"], 0.0)
                self.assertLessEqual(row["target_r_s"], 1.0)
                self.assertGreaterEqual(row["target_q_s"], 0.0)
                self.assertLessEqual(row["target_q_s"], 1.0)
                self.assertEqual(
                    row["actual_history_exposure"] + row["actual_current_exposure"],
                    row["history_budget"] + row["current_budget"],
                )
            self.assertIn("f1", metrics)


if __name__ == "__main__":
    unittest.main()
