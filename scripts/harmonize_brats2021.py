#!/usr/bin/env python3
"""Step 4a: add NET labels to BraTS 2021 using the step-3 four-channel model."""

import argparse
import csv
from pathlib import Path

from brain_tumor_segmentation.data import discover_subjects
from brain_tumor_segmentation.harmonization import extract_brats2021_net, write_harmonized_subject
from brain_tumor_segmentation.model import DEFAULT_INPUT_SHAPE, build_pau_net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Extracted BraTS 2021 root")
    parser.add_argument("weights", type=Path, help="Step-3 four-channel PAU-Net weights")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = build_pau_net(
        DEFAULT_INPUT_SHAPE, 4, "preact", ("NCR", "ED", "NET", "ET")
    )
    model.load_weights(args.weights)
    subjects = discover_subjects(args.dataset, "_seg.nii.gz")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "net_voxel_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "net_voxels"))
        for index, subject in enumerate(subjects, start=1):
            print(f"[{index}/{len(subjects)}] {subject.subject_id}")
            labels, net_voxels = extract_brats2021_net(subject, model)
            write_harmonized_subject(subject, args.output, labels)
            writer.writerow((subject.subject_id, net_voxels))


if __name__ == "__main__":
    main()
