"""Pure validation-history patience logic for an external training monitor.

This module never imports the trainer, changes a checkpoint, or stops a process.
Feed only completed Moon_5 epochs, using summary.json's completed_epochs as the
history boundary. Best-checkpoint selection remains independent of this policy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


class PatienceTracker:
    """Stop after consecutive epochs without an anchor-relative improvement.

    The first eligible epoch establishes an anchor. Smaller gains accumulate
    relative to that unchanged anchor; reaching anchor + min_delta resets the
    counter. Repeated observations of an identical epoch/score are idempotent.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0005, start_epoch: int = 4):
        if type(patience) is not int or patience < 0:
            raise ValueError("patience must be a nonnegative integer")
        if type(start_epoch) is not int or start_epoch < 1:
            raise ValueError("start_epoch must be a positive integer")
        if isinstance(min_delta, bool) or not isinstance(min_delta, (float, int)):
            raise ValueError("min_delta must be a finite nonnegative number")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be a finite nonnegative number")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.start_epoch = start_epoch
        self.anchor_score: float | None = None
        self.anchor_epoch: int | None = None
        self.bad_epochs = 0
        self.last_epoch = 0
        self.stopped = False
        self.stop_epoch: int | None = None
        self.stop_reason: str | None = None
        self.last_event: str | None = None
        self._scores: dict[int, float] = {}

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    @property
    def policy(self) -> dict:
        return {"patience": self.patience, "min_delta": self.min_delta,
                "start_epoch": self.start_epoch, "metric": "Moon_5_miou",
                "threshold_comparison": "score >= anchor + min_delta"}

    def update(self, epoch: int, miou: float) -> bool:
        """Consume one completed validation epoch and return the stop decision."""
        if type(epoch) is not int or epoch < 1:
            raise ValueError("epoch must be a positive integer")
        if isinstance(miou, bool) or not isinstance(miou, (int, float)):
            raise ValueError("miou must be a finite fraction in [0, 1]")
        score = float(miou)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("miou must be a finite fraction in [0, 1]")
        if epoch in self._scores:
            if self._scores[epoch] != score:
                raise ValueError(f"Epoch {epoch} was already observed with a different mIoU")
            self.last_event = "duplicate"
            return self.stopped
        if epoch <= self.last_epoch:
            raise ValueError("Previously unobserved epochs must arrive in increasing order")
        if self.enabled and epoch > max(self.last_epoch + 1, self.start_epoch):
            raise ValueError("Missing eligible validation epochs; replay complete history before proceeding")
        self._scores[epoch] = score
        self.last_epoch = epoch
        if not self.enabled:
            self.last_event = "disabled"
        elif self.stopped:
            self.last_event = "already_stopped"
        elif epoch < self.start_epoch:
            self.last_event = "before_start_epoch"
        elif self.anchor_score is None:
            self.anchor_score, self.anchor_epoch = score, epoch
            self.last_event = "anchor_initialized"
        else:
            threshold = self.anchor_score + self.min_delta
            # A tiny tolerance makes the stated decimal threshold inclusive
            # despite binary representations such as 0.8 + 0.001.
            significant = score >= threshold or math.isclose(score, threshold, rel_tol=0.0, abs_tol=1e-12)
            if significant:
                self.anchor_score, self.anchor_epoch = score, epoch
                self.bad_epochs = 0
                self.last_event = "significant_improvement"
            else:
                self.bad_epochs += 1
                self.last_event = "no_significant_improvement"
                if self.bad_epochs >= self.patience:
                    self.stopped = True
                    self.stop_epoch = epoch
                    self.stop_reason = (f"Moon_5 mIoU failed to improve by at least {self.min_delta:g} "
                                        f"above anchor {self.anchor_score:.12g} for {self.patience} "
                                        f"consecutive eligible epochs; stop at completed epoch {epoch}")
        return self.stopped

    observe = update

    def state_dict(self) -> dict:
        return {"schema": "lusnar_patience_tracker_v1", "policy": self.policy,
                "anchor_score": self.anchor_score, "anchor_epoch": self.anchor_epoch,
                "bad_epochs": self.bad_epochs, "last_epoch": self.last_epoch,
                "stopped": self.stopped, "stop_epoch": self.stop_epoch,
                "stop_reason": self.stop_reason, "last_event": self.last_event,
                "observations": [{"epoch": epoch, "miou": score} for epoch, score in sorted(self._scores.items())]}

    def load_state_dict(self, state: dict) -> None:
        """Restore only the same policy, checking the state against its history."""
        if state.get("schema") != "lusnar_patience_tracker_v1" or state.get("policy") != self.policy:
            raise ValueError("Patience state has a different policy; replay history under the new explicit policy")
        rebuilt = type(self)(self.patience, self.min_delta, self.start_epoch)
        rebuilt.replay(state.get("observations", []), completed_epochs=state["last_epoch"])
        fields = ("anchor_score", "anchor_epoch", "bad_epochs", "last_epoch", "stopped", "stop_epoch", "stop_reason")
        if any(getattr(rebuilt, field) != state.get(field) for field in fields):
            raise ValueError("Patience state is inconsistent with its recorded observations")
        self.__dict__.update(rebuilt.__dict__)
        self.last_event = state.get("last_event")

    @classmethod
    def from_state_dict(cls, state: dict) -> "PatienceTracker":
        policy = state["policy"]
        tracker = cls(policy["patience"], policy["min_delta"], policy["start_epoch"])
        tracker.load_state_dict(state)
        return tracker

    def replay(self, records: Iterable[dict], completed_epochs: int) -> bool:
        """Consume history up to summary.completed_epochs, never an in-flight row.

        Accepts compact {epoch, miou} observations or the trainer's JSONL records
        containing validation.scenes.Moon_5.miou. Calling this repeatedly with
        the full history is safe. All eligible completed epochs must be present.
        """
        if type(completed_epochs) is not int or completed_epochs < 0:
            raise ValueError("completed_epochs must be a nonnegative integer")
        if completed_epochs < self.last_epoch:
            raise ValueError("Completed epoch boundary moved backwards")
        selected = []
        for row in records:
            if type(row.get("epoch")) is not int:
                raise ValueError("History epoch must be an integer")
            if row["epoch"] <= completed_epochs:
                selected.append(row)
        for row in sorted(selected, key=lambda record: record["epoch"]):
            if "miou" in row:
                score = row["miou"]
            else:
                try:
                    score = row["validation"]["scenes"]["Moon_5"]["miou"]
                except (KeyError, TypeError) as error:
                    raise ValueError("History must provide the Moon_5 validation mIoU") from error
            self.update(row["epoch"], score)
        if self.enabled and completed_epochs >= self.start_epoch and self.last_epoch != completed_epochs:
            raise ValueError("History is missing the summary's latest completed validation epoch")
        return self.stopped

    def replay_jsonl(self, path: str | Path, completed_epochs: int) -> bool:
        """Read JSONL without treating an unfinished last write as an epoch."""
        text = Path(path).read_text()
        lines = text.splitlines(keepends=True)
        rows = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not line.endswith("\n"):
                    break
                raise
        return self.replay(rows, completed_epochs)
