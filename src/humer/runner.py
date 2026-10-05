"""Final HUMER training pipeline.

This module implements the released E3-C configuration only: code-based
curriculum construction, adaptive historical review with original persistent
errors, and mandatory full-data knowledge consolidation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import DataLoader

from .config import HumerConfig
from .data import CodeSample, HuggingFaceCodeDataset, collate_code_batch, iter_raw_samples
from .difficulty import DifficultyRecord, curriculum_buckets, load_or_compute_difficulty, split_indices
from .metrics import binary_metrics, safe_div
from .models import (
    build_codebert_factory,
    build_encoder_model_factory,
    build_epvd_factory,
    build_linevul_factory,
    build_pretrained_tokenizer,
)
from .review import CyclicSampler, adaptive_history_ratio, label_stratified_shuffle, stage_slot_schedule
from .utils import set_random_seed


class HumerRunner:
    """Train and evaluate one finalized HUMER configuration."""

    def __init__(self, config: HumerConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        if self.output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True)
        self.log_path = self.output_dir / "training.log"
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.total_optimizer_steps = 0
        self.eval_step_schedule: list[int] = []
        self.error_count: Counter[int] = Counter()
        self.error_rows: list[dict[str, Any]] = []
        self.stage_rows: list[dict[str, Any]] = []
        self.consolidation_rows: list[dict[str, Any]] = []
        self.stage_predictions: dict[int, dict[int, bool]] = {}
        self.selected_checkpoint: str | None = None
        self.selected_validation_metrics: dict[str, float | int] = {}
        self.selected_consolidation_epoch: int | None = None
        self.difficulty_info: dict[str, Any] = {}
        self.epvd_path_stats: dict[str, int] = {}

    def run(self) -> dict[str, Any]:
        set_random_seed(self.config.seed, strict_determinism=self.config.strict_determinism)
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        self._write_config(started_at)
        self._log(f"started_at={started_at.isoformat()} seed={self.config.seed} model={self.config.model}")
        self._build_data_and_model()
        model = self._run_curriculum()
        model = self._run_consolidation(model)

        # The test set is intentionally evaluated exactly once, after all model selection is complete.
        final_metrics = self._evaluate(model, self.test_samples, include_curves=True)
        elapsed = time.perf_counter() - started
        final_metrics.update(
            {
                "seed": self.config.seed,
                "model": self.config.model,
                "model_name": self.config.model_name,
                "selection_metric": "validation_f1",
                "selected_checkpoint": self.selected_checkpoint,
                "selected_consolidation_epoch": self.selected_consolidation_epoch,
                "selected_validation_metrics": self.selected_validation_metrics,
                "total_optimizer_steps": self.total_optimizer_steps,
                "eval_step_schedule": self.eval_step_schedule,
                "wall_clock_seconds": elapsed,
                "wall_clock_minutes": elapsed / 60.0,
                "peak_gpu_memory_bytes": self._peak_gpu_memory(),
                "difficulty": self.difficulty_info,
                "epvd_path_stats": self.epvd_path_stats,
            }
        )
        (self.output_dir / "final_test_metrics.json").write_text(
            json.dumps(final_metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._write_csv(self.output_dir / "stage_diagnostics.csv", self.stage_rows)
        self._write_csv(self.output_dir / "error_tracking.csv", self.error_rows)
        self._write_csv(self.output_dir / "consolidation_history.csv", self.consolidation_rows)
        self._log(
            f"final_test accuracy={final_metrics['accuracy']:.6f} precision={final_metrics['precision']:.6f} "
            f"recall={final_metrics['recall']:.6f} f1={final_metrics['f1']:.6f} "
            f"macro_f1={final_metrics['macro_f1']:.6f} mcc={final_metrics['mcc']:.6f}"
        )
        return final_metrics

    def _build_data_and_model(self) -> None:
        config = self.config
        data_dir = Path(config.data_dir)
        required = [data_dir / name for name in ("train.json", "val.json", "test.json")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing dataset split files: {missing}")

        if config.model == "epvd":
            from .epvd import EpvdPathTokenizer

            tokenizer = EpvdPathTokenizer(
                model_name=str(config.model_name),
                block_size=config.epvd_block_size,
                cache_path=str(config.epvd_path_cache),
                cache_dir=config.hf_cache_dir,
                local_files_only=config.local_files_only,
            )
        else:
            tokenizer = build_pretrained_tokenizer(
                model_type=config.model,
                model_name=str(config.model_name),
                cache_dir=config.hf_cache_dir,
                local_files_only=config.local_files_only,
            )
        self.train_dataset = HuggingFaceCodeDataset(required[0], tokenizer, config.max_length)
        self.val_dataset = HuggingFaceCodeDataset(required[1], tokenizer, config.max_length)
        self.test_dataset = HuggingFaceCodeDataset(required[2], tokenizer, config.max_length)
        self.train_samples = list(iter_raw_samples(self.train_dataset))
        self.val_samples = list(iter_raw_samples(self.val_dataset))
        self.test_samples = list(iter_raw_samples(self.test_dataset))
        self._validate_sample_ids()

        if config.model == "epvd":
            self.epvd_path_stats = tokenizer.prepare(
                [*self.train_samples, *self.val_samples, *self.test_samples]
            )
            tokenizer.write_failures(self.output_dir / "epvd_path_failures.json")
            self.model_factory = build_epvd_factory(
                model_name=str(config.model_name),
                block_size=config.epvd_block_size,
                cnn_size=config.epvd_cnn_size,
                filter_size=config.epvd_filter_size,
                d_size=config.epvd_d_size,
                cache_dir=config.hf_cache_dir,
                local_files_only=config.local_files_only,
            )
        elif config.model == "linevul":
            self.model_factory = build_linevul_factory(
                model_name=str(config.model_name),
                cache_dir=config.hf_cache_dir,
                local_files_only=config.local_files_only,
            )
        elif config.model in {"codebert", "vulgpt"}:
            self.model_factory = build_codebert_factory(
                model_name=str(config.model_name),
                cache_dir=config.hf_cache_dir,
                local_files_only=config.local_files_only,
            )
        else:
            self.model_factory = build_encoder_model_factory(
                model_type=config.model,
                model_name=str(config.model_name),
                cache_dir=config.hf_cache_dir,
                local_files_only=config.local_files_only,
            )

        difficulty_started = time.perf_counter()
        records, cache_path, cache_status = load_or_compute_difficulty(
            self.train_samples, required[0], config.difficulty_cache_dir
        )
        self.train_buckets = [
            [record.sample for record in bucket]
            for bucket in curriculum_buckets(records, config.num_buckets)
        ]
        self.validation_buckets = self._make_validation_buckets()
        self.difficulty_info = {
            "type": "code",
            "cache_path": str(cache_path),
            "cache_status": cache_status,
            "wall_clock_seconds_including_cache_io": time.perf_counter() - difficulty_started,
        }
        self._write_bucket_file(records)
        self._log(
            f"dataset train={len(self.train_samples)} validation={len(self.val_samples)} "
            f"test={len(self.test_samples)} difficulty_cache={cache_status}"
        )

    def _run_curriculum(self):
        model = self.model_factory.create().to(self.device)
        previous_state: dict[str, torch.Tensor] | None = None
        seen_samples: list[CodeSample] = []

        for stage_id, current_bucket in enumerate(self.train_buckets, start=1):
            if previous_state is not None:
                model.load_state_dict(previous_state)
            bucket_steps = math.ceil(len(current_bucket) / self.config.batch_size)
            optimizer = self._create_optimizer(model)
            best_state = clone_state_dict(model)
            best_metrics = self._evaluate(model, self.val_samples)
            selected_phase = "stage_start"
            stage_start_step = self.total_optimizer_steps
            l_hist_before = l_hist_after = l_new_after = forgetting = history_need = new_need = 0.0
            raw_r_s = q_s = persistent_rate = 0.0
            persistent_candidates: list[int] = []
            train_stats: Counter[str] = Counter()

            if stage_id == 1:
                train_stats.update(
                    self._train_plain(
                        model, optimizer, current_bucket, steps_per_epoch=bucket_steps, epochs=3
                    )
                )
                metrics = self._evaluate(model, self.val_samples)
                self.eval_step_schedule.append(self.total_optimizer_steps)
                if is_better_f1(metrics, best_metrics):
                    best_state, best_metrics, selected_phase = clone_state_dict(model), metrics, "stage_training"
            else:
                historical_validation = [
                    sample for bucket in self.validation_buckets[: stage_id - 1] for sample in bucket
                ]
                l_hist_before = self._evaluate_loss(model, historical_validation)
                train_stats.update(
                    self._train_plain(
                        model, optimizer, current_bucket, steps_per_epoch=bucket_steps, epochs=1
                    )
                )
                prelearn_metrics = self._evaluate(model, self.val_samples)
                self.eval_step_schedule.append(self.total_optimizer_steps)
                if is_better_f1(prelearn_metrics, best_metrics):
                    best_state = clone_state_dict(model)
                    best_metrics = prelearn_metrics
                    selected_phase = "prelearn"

                l_hist_after = self._evaluate_loss(model, historical_validation)
                l_new_after = self._evaluate_loss(model, self.validation_buckets[stage_id - 1])
                forgetting = max(0.0, l_hist_after - l_hist_before)
                history_need = l_hist_after + forgetting
                new_need = l_new_after
                raw_r_s = adaptive_history_ratio(l_hist_after, l_new_after, forgetting)
                review_steps = 2 * bucket_steps
                persistent_candidates = self._persistent_error_candidates(model, seen_samples)
                persistent_rate = len(persistent_candidates) / max(1, len(seen_samples))
                q_s = persistent_rate
                schedule, budgets = stage_slot_schedule(
                    self.config.batch_size, review_steps, raw_r_s, q_s
                )
                selected_error_ids = persistent_candidates[: budgets["error_budget"]]

                def evaluate_review(epoch: int) -> None:
                    nonlocal best_state, best_metrics, selected_phase
                    metrics = self._evaluate(model, self.val_samples)
                    self.eval_step_schedule.append(self.total_optimizer_steps)
                    if is_better_f1(metrics, best_metrics):
                        best_state = clone_state_dict(model)
                        best_metrics = metrics
                        selected_phase = f"review_epoch_{epoch}"

                train_stats.update(
                    self._train_review(
                        model=model,
                        optimizer=optimizer,
                        stage_id=stage_id,
                        current_bucket=current_bucket,
                        schedule=schedule,
                        budgets=budgets,
                        selected_error_ids=selected_error_ids,
                        on_epoch_end=evaluate_review,
                        bucket_steps=bucket_steps,
                    )
                )

            checkpoint = self.output_dir / f"stage_{stage_id}_best.pt"
            model.load_state_dict(best_state)
            torch.save(best_state, checkpoint)
            previous_state = clone_state_dict(model)
            seen_samples.extend(current_bucket)
            self._update_error_counts(model, stage_id, seen_samples)
            self.stage_rows.append(
                {
                    "stage_id": stage_id,
                    "current_bucket_id": stage_id,
                    "current_bucket_size": len(current_bucket),
                    "historical_sample_count": len(seen_samples) - len(current_bucket),
                    "L_hist_before": l_hist_before,
                    "L_hist_after": l_hist_after,
                    "L_new_after": l_new_after,
                    "F_s": forgetting,
                    "H_s": history_need,
                    "U_s": new_need,
                    "raw_r_s": raw_r_s,
                    "target_r_s": raw_r_s,
                    "effective_r_s": train_stats.get("effective_r_s", 0.0),
                    "persistent_error_candidate_count": len(persistent_candidates),
                    "persistent_error_rate": persistent_rate,
                    "target_q_s": q_s,
                    "effective_q_s": train_stats.get("effective_q_s", 0.0),
                    "history_budget": train_stats.get("history_budget", 0),
                    "current_budget": train_stats.get("current_budget", 0),
                    "target_error_budget": train_stats.get("target_error_budget", 0),
                    "actual_history_exposure": train_stats.get("actual_history_exposure", 0),
                    "actual_current_exposure": train_stats.get("actual_current_exposure", 0),
                    "actual_targeted_error_exposure": train_stats.get("error_exposure", 0),
                    "actual_general_history_exposure": train_stats.get("general_exposure", 0),
                    "unique_error_samples_replayed": train_stats.get("unique_error_samples", 0),
                    "history_slots_per_batch_min": train_stats.get("history_slots_min", 0),
                    "history_slots_per_batch_max": train_stats.get("history_slots_max", 0),
                    "error_slots_per_batch_min": train_stats.get("error_slots_min", 0),
                    "error_slots_per_batch_max": train_stats.get("error_slots_max", 0),
                    "optimizer_steps_at_stage_start": stage_start_step,
                    "optimizer_steps_at_stage_end": self.total_optimizer_steps,
                    "validation_f1": best_metrics["f1"],
                    "validation_macro_f1": best_metrics["macro_f1"],
                    "validation_mcc": best_metrics["mcc"],
                    "selected_phase": selected_phase,
                    "selected_checkpoint": str(checkpoint),
                }
            )
            self._log(
                f"stage={stage_id} validation_f1={best_metrics['f1']:.6f} "
                f"selected_phase={selected_phase} checkpoint={checkpoint}"
            )
        if previous_state is None:
            raise RuntimeError("Curriculum training did not produce a Stage checkpoint.")
        model.load_state_dict(previous_state)
        return model

    def _run_consolidation(self, model):
        """Select the final model only from post-consolidation checkpoints."""

        steps_per_epoch = math.ceil(len(self.train_samples) / self.config.batch_size)
        optimizer = self._create_optimizer(model)
        best_state: dict[str, torch.Tensor] | None = None
        best_metrics: dict[str, float | int] | None = None
        best_epoch: int | None = None
        stale_epochs = 0
        checkpoint = self.output_dir / "consolidation_best.pt"

        for epoch in range(1, self.config.consolidation_max_epochs + 1):
            self._train_plain(model, optimizer, self.train_samples, steps_per_epoch, epochs=1)
            validation_metrics = self._evaluate(model, self.val_samples)
            self.eval_step_schedule.append(self.total_optimizer_steps)
            improved = best_metrics is None or is_better_f1(validation_metrics, best_metrics)
            self.consolidation_rows.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": self.total_optimizer_steps,
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_precision": validation_metrics["precision"],
                    "validation_recall": validation_metrics["recall"],
                    "validation_f1": validation_metrics["f1"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "validation_mcc": validation_metrics["mcc"],
                    "is_best_so_far": int(improved),
                }
            )
            self._log(
                f"consolidation epoch={epoch} validation_f1={validation_metrics['f1']:.6f} "
                f"validation_macro_f1={validation_metrics['macro_f1']:.6f}"
            )
            if improved:
                best_state = clone_state_dict(model)
                best_metrics = validation_metrics
                best_epoch = epoch
                torch.save(best_state, checkpoint)
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.config.consolidation_patience:
                break

        if best_state is None or best_metrics is None or best_epoch is None:
            raise RuntimeError("Knowledge consolidation did not produce a checkpoint.")
        model.load_state_dict(best_state)
        torch.save(best_state, self.output_dir / "final_model.pt")
        self.selected_checkpoint = str(checkpoint)
        self.selected_consolidation_epoch = best_epoch
        self.selected_validation_metrics = best_metrics
        for row in self.consolidation_rows:
            row["selected_for_final"] = int(int(row["epoch"]) == best_epoch)
        self._log(
            f"consolidation selected_epoch={best_epoch} validation_f1={best_metrics['f1']:.6f} "
            f"checkpoint={checkpoint}"
        )
        return model

    def _train_plain(
        self,
        model,
        optimizer,
        samples: list[CodeSample],
        steps_per_epoch: int,
        epochs: int,
    ) -> Counter[str]:
        sampler = CyclicSampler(
            samples, random.Random(self.config.seed + self.total_optimizer_steps + 11)
        )
        counts: Counter[str] = Counter()
        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                batch = sampler.take(self.config.batch_size)
                self._train_batch(model, optimizer, batch)
                counts["current"] += len(batch)
        return counts

    def _train_review(
        self,
        model,
        optimizer,
        stage_id: int,
        current_bucket: list[CodeSample],
        schedule: list[tuple[int, int, int]],
        budgets: dict[str, int],
        selected_error_ids: list[int],
        on_epoch_end: Callable[[int], None],
        bucket_steps: int,
    ) -> Counter[str]:
        rng = random.Random(self.config.seed + self.total_optimizer_steps + 101)
        historical_buckets = self.train_buckets[: stage_id - 1]
        by_id = {sample.sample_id: sample for bucket in historical_buckets for sample in bucket}
        error_samples = [by_id[sample_id] for sample_id in selected_error_ids if sample_id in by_id]
        excluded = set(selected_error_ids)
        general_buckets = [
            [sample for sample in bucket if sample.sample_id not in excluded]
            for bucket in historical_buckets
        ]
        general_sequence = build_general_history_sequence(
            general_buckets, budgets["general_history_budget"], rng
        )
        error_sequence = repeated_sequence(error_samples, budgets["error_budget"], rng)
        current_sequence = repeated_sequence(current_bucket, budgets["current_budget"], rng)
        current_offset = general_offset = error_offset = 0
        replayed_error_ids: set[int] = set()
        counts: Counter[str] = Counter()
        history_slots: list[int] = []
        error_slots: list[int] = []

        for index, (current_count, general_count, error_count) in enumerate(schedule):
            current_part = current_sequence[current_offset : current_offset + current_count]
            general_part = general_sequence[general_offset : general_offset + general_count]
            error_part = error_sequence[error_offset : error_offset + error_count]
            current_offset += current_count
            general_offset += general_count
            error_offset += error_count
            replayed_error_ids.update(sample.sample_id for sample in error_part)
            sourced = [
                *[(sample, "current") for sample in current_part],
                *[(sample, "general") for sample in general_part],
                *[(sample, "error") for sample in error_part],
            ]
            rng.shuffle(sourced)
            self._train_batch(model, optimizer, [sample for sample, _ in sourced])
            counts["current"] += len(current_part)
            counts["general"] += len(general_part)
            counts["error"] += len(error_part)
            history_slots.append(general_count + error_count)
            error_slots.append(error_count)
            if (index + 1) % bucket_steps == 0:
                on_epoch_end((index + 1) // bucket_steps)

        counts["history_budget"] = budgets["history_budget"]
        counts["current_budget"] = budgets["current_budget"]
        counts["target_error_budget"] = budgets["error_budget"]
        counts["actual_history_exposure"] = counts["general"] + counts["error"]
        counts["actual_current_exposure"] = counts["current"]
        counts["general_exposure"] = counts["general"]
        counts["error_exposure"] = counts["error"]
        counts["unique_error_samples"] = len(replayed_error_ids)
        counts["effective_r_s"] = safe_div(
            counts["actual_history_exposure"], budgets["total_review_positions"]
        )
        counts["effective_q_s"] = safe_div(counts["error"], counts["actual_history_exposure"])
        counts["history_slots_min"] = min(history_slots, default=0)
        counts["history_slots_max"] = max(history_slots, default=0)
        counts["error_slots_min"] = min(error_slots, default=0)
        counts["error_slots_max"] = max(error_slots, default=0)
        return counts

    def _train_batch(self, model, optimizer, samples: list[CodeSample]) -> None:
        if self.config.model == "epvd":
            self._train_epvd_batch(model, optimizer, samples)
            return
        dataset = self.train_dataset.with_samples(samples)
        batch = collate_code_batch([dataset[index] for index in range(len(dataset))])
        batch = move_batch(batch, self.device)
        model.train()
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = output["loss"] if isinstance(output, dict) else output.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.total_optimizer_steps += 1

    def _train_epvd_batch(self, model, optimizer, samples: list[CodeSample]) -> None:
        micro_size = self.config.epvd_micro_batch_size
        if len(samples) % micro_size:
            raise ValueError("EPVD logical batches must be divisible by epvd_micro_batch_size.")
        micro_batches = [samples[offset : offset + micro_size] for offset in range(0, len(samples), micro_size)]
        optimizer.zero_grad()
        model.train()
        for micro_samples in micro_batches:
            dataset = self.train_dataset.with_samples(micro_samples)
            batch = move_batch(
                collate_code_batch([dataset[index] for index in range(len(dataset))]), self.device
            )
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            (output["loss"] / len(micro_batches)).backward()
        optimizer.step()
        self.total_optimizer_steps += 1

    def _evaluate_loss(self, model, samples: Sequence[CodeSample]) -> float:
        if not samples:
            return 0.0
        dataset = self.val_dataset.with_samples(samples)
        loader = DataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=collate_code_batch,
        )
        total_loss = 0.0
        total_samples = 0
        model.eval()
        with torch.no_grad():
            for raw_batch in loader:
                batch = move_batch(raw_batch, self.device)
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = output["loss"] if isinstance(output, dict) else output.loss
                count = int(batch["labels"].shape[0])
                total_loss += float(loss.item()) * count
                total_samples += count
        return total_loss / max(1, total_samples)

    def _evaluate(
        self, model, samples: Sequence[CodeSample], include_curves: bool = False
    ) -> dict[str, float | int]:
        dataset = self.val_dataset.with_samples(samples)
        loader = DataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=collate_code_batch,
        )
        labels: list[int] = []
        probabilities: list[float] = []
        model.eval()
        with torch.no_grad():
            for raw_batch in loader:
                batch = move_batch(raw_batch, self.device)
                logits = get_logits(model, batch)
                probabilities.extend(
                    float(value) for value in torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                )
                labels.extend(int(value) for value in batch["labels"].cpu().tolist())
        predictions = [
            int(probability >= self.config.decision_threshold) for probability in probabilities
        ]
        metrics = binary_metrics(labels, predictions, probabilities)
        if not include_curves:
            metrics.pop("roc_auc", None)
            metrics.pop("auprc", None)
        return metrics

    def _persistent_error_candidates(self, model, historical_samples: list[CodeSample]) -> list[int]:
        ids = [
            sample.sample_id
            for sample in historical_samples
            if self.error_count[sample.sample_id] >= 2
        ]
        if not ids:
            ids = [
                sample.sample_id
                for sample in historical_samples
                if self.error_count[sample.sample_id] >= 1
            ]
        probabilities = self._true_label_probabilities(model, historical_samples)
        ids.sort(key=lambda sample_id: (-self.error_count[sample_id], probabilities[sample_id], sample_id))
        return ids

    def _true_label_probabilities(
        self, model, samples: Sequence[CodeSample]
    ) -> dict[int, float]:
        dataset = self.train_dataset.with_samples(samples)
        loader = DataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=collate_code_batch,
        )
        result: dict[int, float] = {}
        offset = 0
        model.eval()
        with torch.no_grad():
            for raw_batch in loader:
                batch = move_batch(raw_batch, self.device)
                probabilities = torch.softmax(get_logits(model, batch), dim=-1).cpu()
                labels = batch["labels"].cpu().tolist()
                for local_index, label in enumerate(labels):
                    sample_id = samples[offset + local_index].sample_id
                    result[sample_id] = float(probabilities[local_index, int(label)].item())
                offset += len(labels)
        return result

    def _update_error_counts(
        self, model, stage_id: int, seen_samples: Sequence[CodeSample]
    ) -> None:
        correctness = self._predict_correctness(model, seen_samples)
        self.stage_predictions[stage_id] = correctness
        for sample in seen_samples:
            correct = correctness[sample.sample_id]
            if not correct:
                self.error_count[sample.sample_id] += 1
            self.error_rows.append(
                {
                    "stage_id": stage_id,
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "correct": int(correct),
                    "error_count": self.error_count[sample.sample_id],
                    "persistent_error": int(self.error_count[sample.sample_id] >= 2),
                }
            )

    def _predict_correctness(
        self, model, samples: Sequence[CodeSample]
    ) -> dict[int, bool]:
        dataset = self.train_dataset.with_samples(samples)
        loader = DataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=collate_code_batch,
        )
        result: dict[int, bool] = {}
        offset = 0
        model.eval()
        with torch.no_grad():
            for raw_batch in loader:
                batch = move_batch(raw_batch, self.device)
                predictions = get_logits(model, batch).argmax(dim=-1).cpu().tolist()
                labels = batch["labels"].cpu().tolist()
                for local_index, (prediction, label) in enumerate(zip(predictions, labels)):
                    result[samples[offset + local_index].sample_id] = int(prediction) == int(label)
                offset += len(labels)
        return result

    def _make_validation_buckets(self) -> list[list[CodeSample]]:
        by_label: dict[int, list[CodeSample]] = defaultdict(list)
        for sample in self.val_samples:
            by_label[int(sample.label)].append(sample)
        buckets: list[list[CodeSample]] = [[] for _ in range(self.config.num_buckets)]
        for label, samples in sorted(by_label.items()):
            ordered = sorted(samples, key=lambda sample: sample.sample_id)
            for bucket_id, indices in enumerate(split_indices(len(ordered), self.config.num_buckets)):
                buckets[bucket_id].extend(ordered[index] for index in indices)
        return [sorted(bucket, key=lambda sample: (sample.label, sample.sample_id)) for bucket in buckets]

    def _create_optimizer(self, model):
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _validate_sample_ids(self) -> None:
        for split_name, samples in (
            ("train", self.train_samples),
            ("validation", self.val_samples),
            ("test", self.test_samples),
        ):
            ids = [sample.sample_id for sample in samples]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate sample IDs found in the {split_name} split.")

    def _write_config(self, started_at: datetime) -> None:
        payload = {
            "created_at": started_at.isoformat(),
            "algorithm": {
                "difficulty": "code",
                "bucket_strategy": "label_stratified",
                "curriculum_order": "easy_to_hard",
                "stage_1_epochs": 3,
                "prelearn_epochs": 1,
                "review_epochs": 2,
                "history_ratio": "H_s / (H_s + U_s + 1e-8), no clipping",
                "error_ratio": "persistent_error_count / historical_sample_count",
                "checkpoint_metric": "validation_f1",
                "final_candidates": "post_consolidation_checkpoints_only",
                "scheduler": None,
            },
            "runtime": self.config.as_dict(),
        }
        (self.output_dir / "config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _write_bucket_file(self, records: Sequence[DifficultyRecord]) -> None:
        score_by_id = {record.sample.sample_id: record.score for record in records}
        rows = []
        for bucket_id, bucket in enumerate(self.train_buckets, start=1):
            for sample in bucket:
                rows.append(
                    {
                        "bucket_id": bucket_id,
                        "sample_id": sample.sample_id,
                        "label": sample.label,
                        "difficulty": score_by_id[sample.sample_id],
                    }
                )
        self._write_csv(self.output_dir / "difficulty_buckets.csv", rows)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _peak_gpu_memory(self) -> int:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            return int(torch.cuda.max_memory_allocated(self.device))
        return 0

    def _log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def build_general_history_sequence(
    buckets: Sequence[Sequence[CodeSample]], budget: int, rng: random.Random
) -> list[CodeSample]:
    nonempty = [list(bucket) for bucket in buckets if bucket]
    if budget == 0:
        return []
    if not nonempty:
        raise ValueError("General-history budget is positive, but no general-history samples exist.")
    base, remainder = divmod(budget, len(nonempty))
    per_bucket = [base + (1 if index < remainder else 0) for index in range(len(nonempty))]
    sequences = [
        stratified_repeated_take(bucket, quota, rng)
        for bucket, quota in zip(nonempty, per_bucket)
    ]
    result: list[CodeSample] = []
    offsets = [0] * len(sequences)
    while len(result) < budget:
        for index, sequence in enumerate(sequences):
            if offsets[index] < len(sequence):
                result.append(sequence[offsets[index]])
                offsets[index] += 1
    return result


def stratified_repeated_take(
    samples: Sequence[CodeSample], quota: int, rng: random.Random
) -> list[CodeSample]:
    if quota == 0:
        return []
    by_label: dict[int, list[CodeSample]] = defaultdict(list)
    for sample in samples:
        by_label[int(sample.label)].append(sample)
    labels = sorted(by_label)
    raw = {label: quota * len(by_label[label]) / len(samples) for label in labels}
    counts = {label: math.floor(raw[label]) for label in labels}
    remaining = quota - sum(counts.values())
    order = sorted(labels, key=lambda label: (-(raw[label] - counts[label]), label))
    for label in order[:remaining]:
        counts[label] += 1
    selected: list[CodeSample] = []
    for label in labels:
        sampler = CyclicSampler(
            label_stratified_shuffle(by_label[label], rng), rng
        )
        selected.extend(sampler.take(counts[label]))
    rng.shuffle(selected)
    return selected


def repeated_sequence(
    samples: Sequence[CodeSample], count: int, rng: random.Random
) -> list[CodeSample]:
    if count == 0:
        return []
    if not samples:
        raise ValueError("A positive replay budget requires at least one source sample.")
    return CyclicSampler(samples, rng).take(count)


def clone_state_dict(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def get_logits(model, batch: dict[str, Any]):
    output = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=None
    )
    return output["logits"] if isinstance(output, dict) else output.logits


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def is_better_f1(
    metrics: dict[str, float | int], best: dict[str, float | int]
) -> bool:
    current_f1 = float(metrics["f1"])
    best_f1 = float(best["f1"])
    if current_f1 != best_f1:
        return current_f1 > best_f1
    return float(metrics["macro_f1"]) > float(best["macro_f1"])
