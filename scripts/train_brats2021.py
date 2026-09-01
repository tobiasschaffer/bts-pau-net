#!/usr/bin/env python3
"""Step 1: train the three-region PAU-Net on BraTS 2021."""

import argparse
from pathlib import Path

from brain_tumor_segmentation.labels import brats2021_regions
from brain_tumor_segmentation.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Extracted BraTS 2021 training-data root")
    parser.add_argument("output", type=Path, help="Directory for checkpoints and logs")
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--initial-weights", type=Path)
    args = parser.parse_args()
    train_model(
        [args.dataset], args.output, "_seg.nii.gz", brats2021_regions,
        ("ET", "TC", "WT"), args.train_count, args.validation_fraction,
        args.epochs, args.batch_size, args.seed, args.initial_weights,
    )


if __name__ == "__main__":
    main()
