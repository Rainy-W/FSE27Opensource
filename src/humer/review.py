"""Deterministic stage-level allocation for adaptive historical review."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

from .data import CodeSample


def adaptive_history_ratio(historical_loss: float, new_loss: float, forgetting: float) -> float:
    history_need = historical_loss + forgetting
    return history_need / (history_need + new_loss + 1e-8)


def stage_slot_schedule(
    batch_size: int, review_steps: int, history_ratio: float, error_ratio: float
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    """Return ``(current, general_history, persistent_error)`` for every update."""

    total_positions = batch_size * review_steps
    history_budget = round(total_positions * history_ratio)
    error_budget = round(history_budget * error_ratio)
    schedule: list[tuple[int, int, int]] = []
    previous_history = 0
    previous_error = 0
    for step in range(1, review_steps + 1):
        cumulative_history = round(step * batch_size * history_ratio)
        cumulative_error = round(error_ratio * cumulative_history)
        history_slots = cumulative_history - previous_history
        error_slots = cumulative_error - previous_error
        schedule.append((batch_size - history_slots, history_slots - error_slots, error_slots))
        previous_history = cumulative_history
        previous_error = cumulative_error
    actual_current = sum(item[0] for item in schedule)
    actual_general = sum(item[1] for item in schedule)
    actual_error = sum(item[2] for item in schedule)
    expected = (total_positions - history_budget, history_budget - error_budget, error_budget)
    if (actual_current, actual_general, actual_error) != expected:
        raise RuntimeError("Stage-level review allocation does not match its target budget.")
    return schedule, {
        "total_review_positions": total_positions,
        "history_budget": history_budget,
        "current_budget": total_positions - history_budget,
        "error_budget": error_budget,
        "general_history_budget": history_budget - error_budget,
    }


class CyclicSampler:
    """Cycle through a deterministically shuffled sample pool."""

    def __init__(self, samples: Sequence[CodeSample], rng: random.Random) -> None:
        if not samples:
            raise ValueError("CyclicSampler requires at least one sample.")
        self.samples = list(samples)
        self.rng = rng
        self.position = 0
        self._shuffle()

    def _shuffle(self) -> None:
        self.rng.shuffle(self.samples)
        self.position = 0

    def take(self, count: int) -> list[CodeSample]:
        selected: list[CodeSample] = []
        while len(selected) < count:
            if self.position >= len(self.samples):
                self._shuffle()
            width = min(count - len(selected), len(self.samples) - self.position)
            selected.extend(self.samples[self.position : self.position + width])
            self.position += width
        return selected


def label_stratified_shuffle(samples: Sequence[CodeSample], rng: random.Random) -> list[CodeSample]:
    by_label: dict[int, list[CodeSample]] = defaultdict(list)
    for sample in samples:
        by_label[int(sample.label)].append(sample)
    shuffled: list[CodeSample] = []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda sample: sample.sample_id)
        rng.shuffle(group)
        shuffled.extend(group)
    rng.shuffle(shuffled)
    return shuffled
