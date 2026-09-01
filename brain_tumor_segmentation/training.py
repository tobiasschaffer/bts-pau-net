"""Shared command-line training workflow."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import tensorflow as tf

from .data import BraTSSequence, Subject, discover_subjects
from .model import DEFAULT_INPUT_SHAPE, build_pau_net


def train_model(
    dataset_roots: Sequence[Path],
    output_dir: Path,
    label_suffix: str,
    label_transform: Callable,
    channel_names: Sequence[str],
    train_count: int | None,
    validation_fraction: float,
    epochs: int,
    batch_size: int,
    seed: int,
    initial_weights: Path | None = None,
) -> tf.keras.callbacks.History:
    subjects: list[Subject] = []
    for root in dataset_roots:
        subjects.extend(discover_subjects(root, label_suffix))
    random.Random(seed).shuffle(subjects)

    if train_count is None:
        validation_count = max(1, round(len(subjects) * validation_fraction))
        train_count = len(subjects) - validation_count
    if not 0 < train_count < len(subjects):
        raise ValueError(f"train_count must be between 1 and {len(subjects) - 1}")

    training_subjects = subjects[:train_count]
    validation_subjects = subjects[train_count:]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "split", "directory"))
        writer.writerows((item.subject_id, "train", item.directory) for item in training_subjects)
        writer.writerows((item.subject_id, "validation", item.directory) for item in validation_subjects)

    train_sequence = BraTSSequence(training_subjects, DEFAULT_INPUT_SHAPE, label_transform, batch_size, True, seed)
    validation_sequence = BraTSSequence(validation_subjects, DEFAULT_INPUT_SHAPE, label_transform, batch_size)
    model = build_pau_net(DEFAULT_INPUT_SHAPE, len(channel_names), "preact", channel_names)
    if initial_weights is not None:
        model.load_weights(initial_weights)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "model_{epoch:03d}_{val_loss:.4f}.weights.h5",
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "history.csv"),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-3 if epoch < 5 else 1e-4 if epoch < 10 else 1e-5
        ),
    ]
    return model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=epochs,
        callbacks=callbacks,
    )
