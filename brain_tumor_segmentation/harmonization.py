"""NET extraction and conversion to the harmonized four-compartment encoding."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, generate_binary_structure

from .data import Subject, center_pad, load_modalities, read_image
from .labels import brats2021_regions

HIGH_RESOLUTION_SHAPE = (310, 480, 480)


def extract_brats2018_net(subject: Subject, brats2021_model) -> tuple[np.ndarray, int]:
    """Apply the original cross-dataset heuristic to one BraTS 2018 subject."""
    from .model import DEFAULT_INPUT_SHAPE

    if subject.segmentation is None:
        raise FileNotFoundError(f"Missing segmentation for {subject.subject_id}")
    inputs = load_modalities(subject, DEFAULT_INPUT_SHAPE)[None, ...]
    prediction_2021 = brats2021_model.predict(inputs, verbose=0)[0] > 0.5
    reference = brats2021_regions(read_image(subject.segmentation), DEFAULT_INPUT_SHAPE[1:])

    eroded_predicted_tc = binary_erosion(prediction_2021[1], iterations=1)
    net = reference[1] & ~reference[0] & ~eroded_predicted_tc
    ncr = reference[1] & ~reference[0] & ~net
    net = binary_opening(net, iterations=1)
    net = binary_dilation(net, structure=generate_binary_structure(3, 2), iterations=1)
    net &= ~reference[0]
    ncr = binary_opening(ncr, iterations=1)
    et = reference[1] & ~net & ~ncr
    ed = reference[2] & ~reference[1] & ~net & ~ncr
    encoded = ncr.astype(np.uint8) + 2 * ed + 3 * net + 4 * et
    return center_pad(encoded, HIGH_RESOLUTION_SHAPE).astype(np.uint8), int(net.sum())


def extract_brats2021_net(subject: Subject, brats2018_net_model) -> tuple[np.ndarray, int]:
    """Extend a BraTS 2021 reference mask with NET predicted by the step-3 model."""
    from .model import DEFAULT_INPUT_SHAPE

    if subject.segmentation is None:
        raise FileNotFoundError(f"Missing segmentation for {subject.subject_id}")
    inputs = load_modalities(subject, DEFAULT_INPUT_SHAPE)[None, ...]
    prediction = brats2018_net_model.predict(inputs, verbose=0)[0] > 0.5
    net = binary_opening(prediction[2], iterations=1)
    net = binary_dilation(net, structure=generate_binary_structure(3, 2), iterations=1)

    output_shape = tuple(size * 2 for size in DEFAULT_INPUT_SHAPE[1:])
    et, tc, wt = brats2021_regions(read_image(subject.segmentation), output_shape)
    ncr = tc & ~et & ~net
    ed = wt & ~tc & ~et & ~net
    et = et & ~net
    encoded = ncr.astype(np.uint8) + 2 * ed + 3 * net + 4 * et
    return center_pad(encoded, HIGH_RESOLUTION_SHAPE).astype(np.uint8), int(net.sum())


def write_harmonized_subject(
    subject: Subject, output_root: Path, labels: np.ndarray, copy_inputs: bool = True
) -> None:
    target = output_root / subject.directory.name
    target.mkdir(parents=True, exist_ok=True)
    if copy_inputs:
        for path in subject.modalities.values():
            shutil.copy2(path, target / path.name)
        if subject.segmentation:
            shutil.copy2(subject.segmentation, target / subject.segmentation.name)
    highres_image = sitk.GetImageFromArray(labels)
    highres_image.SetOrigin((0.0, -479.0, 0.0))
    sitk.WriteImage(highres_image, str(target / f"{subject.subject_id}_seg_4label_highres.nii.gz"))
    reduced_image = sitk.GetImageFromArray(labels[::2, ::2, ::2])
    reduced_image.SetOrigin((0.0, -239.0, 0.0))
    sitk.WriteImage(reduced_image, str(target / f"{subject.subject_id}_seg_4label.nii.gz"))
