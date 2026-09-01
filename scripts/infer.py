#!/usr/bin/env python3
"""Run the final five-channel PAU-Net and write four-compartment NIfTI masks."""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_opening, generate_binary_structure

from brain_tumor_segmentation.data import center_pad, discover_subjects, load_modalities
from brain_tumor_segmentation.model import DEFAULT_INPUT_SHAPE, build_pau_net


def decode_prediction(prediction: np.ndarray) -> np.ndarray:
    """Convert [ET, TC, NET, TCN, WT] probabilities to labels 0..4."""
    regions = prediction > 0.5
    net = binary_opening(regions[2], iterations=1)
    net = binary_dilation(net, structure=generate_binary_structure(3, 2), iterations=1)
    ncr_raw = regions[1] & ~regions[0] & ~net
    ncr = binary_opening(ncr_raw, iterations=1)
    et = (regions[0] | (ncr_raw & ~ncr)) & ~net
    ed = regions[4] & ~regions[1] & ~regions[0] & ~net
    return (ncr + 2 * ed + 3 * net + 4 * et).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = build_pau_net(
        DEFAULT_INPUT_SHAPE, 5, "preact", ("ET", "TC", "NET", "TCN", "WT")
    )
    model.load_weights(args.weights)
    subjects = discover_subjects(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, subject in enumerate(subjects, start=1):
        print(f"[{index}/{len(subjects)}] {subject.subject_id}")
        prediction = model.predict(load_modalities(subject, DEFAULT_INPUT_SHAPE)[None], verbose=0)[0]
        highres = center_pad(decode_prediction(prediction), (310, 480, 480))
        target = args.output / subject.subject_id
        target.mkdir(parents=True, exist_ok=True)
        highres_image = sitk.GetImageFromArray(highres)
        highres_image.SetOrigin((0.0, -479.0, 0.0))
        sitk.WriteImage(highres_image, str(target / f"{subject.subject_id}_seg_4label_highres.nii.gz"))
        reduced_image = sitk.GetImageFromArray(highres[::2, ::2, ::2])
        reduced_image.SetOrigin((0.0, -239.0, 0.0))
        sitk.WriteImage(reduced_image, str(target / f"{subject.subject_id}_seg_4label.nii.gz"))


if __name__ == "__main__":
    main()
