"""Label mappings used by the successive BraTS experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .data import center_crop


def _crop(mask: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    return center_crop(mask, shape)


def brats2021_regions(labels: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Return [ET, TC, WT] from standard BraTS labels 1, 2, and 4."""
    et = labels == 4
    tc = (labels == 4) | (labels == 1)
    wt = tc | (labels == 2)
    masks = (et, tc, wt)
    if any(target > current for target, current in zip(shape, labels.shape)):
        # The original training notebook cropped the standard-resolution label
        # to the input crop and then used nearest-neighbour 2x upsampling.
        base_shape = tuple(target // 2 for target in shape)
        masks = tuple(
            _crop(mask, base_shape).repeat(2, 0).repeat(2, 1).repeat(2, 2)
            for mask in masks
        )
    else:
        masks = tuple(_crop(mask, shape) for mask in masks)
    return np.asarray(masks, dtype=np.uint8)


def four_compartments(labels: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Return [NCR, ED, NET, ET] from the harmonized 0..4 encoding."""
    return np.asarray([_crop(labels == value, shape) for value in (1, 2, 3, 4)], dtype=np.uint8)


def harmonized_regions(labels: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Return [ET, TC, NET, TCN, WT] for final combined training."""
    et = labels == 4
    ncr = labels == 1
    ed = labels == 2
    net = labels == 3
    tc = et | ncr
    tcn = tc | net
    wt = tcn | ed
    return np.asarray([_crop(mask, shape) for mask in (et, tc, net, tcn, wt)], dtype=np.uint8)
