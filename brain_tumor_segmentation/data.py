"""BraTS NIfTI discovery, preprocessing, and Keras data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

MODALITIES = ("t1", "t2", "t1ce", "flair")


class Subject(NamedTuple):
    subject_id: str
    directory: Path
    modalities: dict[str, Path]
    segmentation: Path | None


def read_image(path: Path | str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def center_crop(array: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    if len(shape) != 3 or any(target > current for target, current in zip(shape, array.shape)):
        raise ValueError(f"Cannot center-crop shape {array.shape} to {tuple(shape)}")
    starts = [(current - target) // 2 for current, target in zip(array.shape, shape)]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, shape))
    return array[slices]


def center_pad(array: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    if len(shape) != 3 or any(target < current for target, current in zip(shape, array.shape)):
        raise ValueError(f"Cannot center-pad shape {array.shape} to {tuple(shape)}")
    widths = []
    for current, target in zip(array.shape, shape):
        before = (target - current) // 2
        widths.append((before, target - current - before))
    return np.pad(array, widths, mode="constant")


def normalize(array: np.ndarray, shape: Sequence[int] | None = None) -> np.ndarray:
    array = center_crop(array, shape) if shape is not None else array
    array = array.astype(np.float32, copy=False)
    std = float(array.std())
    if std == 0.0:
        return np.zeros_like(array, dtype=np.float32)
    return (array - float(array.mean())) / std


def discover_subjects(root: Path | str, label_suffix: str = "_seg.nii.gz") -> list[Subject]:
    """Recursively discover subjects from their four modality files."""
    root = Path(root)
    subjects = []
    for t1_path in sorted(root.rglob("*_t1.nii.gz")):
        subject_id = t1_path.name[: -len("_t1.nii.gz")]
        directory = t1_path.parent
        modalities = {name: directory / f"{subject_id}_{name}.nii.gz" for name in MODALITIES}
        missing = [str(path) for path in modalities.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete subject {subject_id}: {', '.join(missing)}")
        label_path = directory / f"{subject_id}{label_suffix}"
        subjects.append(Subject(subject_id, directory, modalities, label_path if label_path.is_file() else None))
    if not subjects:
        raise FileNotFoundError(f"No BraTS subjects found below {root}")
    return subjects


def load_modalities(subject: Subject, input_shape: Sequence[int]) -> np.ndarray:
    spatial_shape = tuple(input_shape[1:])
    return np.asarray(
        [normalize(read_image(subject.modalities[name]), spatial_shape) for name in MODALITIES],
        dtype=np.float32,
    )


class BraTSSequence(tf.keras.utils.Sequence):
    """Memory-efficient batch loader for the four BraTS MRI modalities."""

    def __init__(
        self,
        subjects: Sequence[Subject],
        input_shape: Sequence[int],
        label_transform: Callable[[np.ndarray, Sequence[int]], np.ndarray],
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 1,
    ):
        super().__init__()
        self.subjects = list(subjects)
        self.input_shape = tuple(input_shape)
        self.label_transform = label_transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.subjects))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.subjects) / self.batch_size))

    def __getitem__(self, batch_index: int):
        selected = self.indices[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        images, labels = [], []
        output_shape = tuple(size * 2 for size in self.input_shape[1:])
        for index in selected:
            subject = self.subjects[index]
            if subject.segmentation is None:
                raise FileNotFoundError(f"No segmentation for {subject.subject_id}")
            images.append(load_modalities(subject, self.input_shape))
            labels.append(self.label_transform(read_image(subject.segmentation), output_shape))
        return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.uint8)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
