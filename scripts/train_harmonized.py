#!/usr/bin/env python3
"""Step 4: train the final five-region PAU-Net on harmonized BraTS 2018/2021."""

import argparse
from pathlib import Path

from brain_tumor_segmentation.labels import harmonized_regions
from brain_tumor_segmentation.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brats2018", type=Path, help="Harmonized BraTS 2018 root")
    parser.add_argument("brats2021", type=Path, help="Harmonized BraTS 2021 root")
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--initial-weights", type=Path)
    args = parser.parse_args()
    train_model(
        [args.brats2018, args.brats2021], args.output,
        "_seg_4label_highres.nii.gz", harmonized_regions,
        ("ET", "TC", "NET", "TCN", "WT"), args.train_count,
        args.validation_fraction, args.epochs, args.batch_size, args.seed,
        args.initial_weights,
    )


if __name__ == "__main__":
    main()
