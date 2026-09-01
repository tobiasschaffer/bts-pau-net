# PAU-Net Brain-Tumor Segmentation for NEHT Compartment

The repository contains the training and inference pipeline used to identify
four brain-tumor compartments in multi-modal MRI data:

| Value | Compartment | Abbreviation |
| ---: | --- | --- |
| 0 | Background | — |
| 1 | Necrotic tumor core | NCR |
| 2 | Peritumoral edema | ED |
| 3 | Non-enhancing tumor | NET/NEHT |
| 4 | Enhancing tumor | ET |

It implements the PAU-Net architecture, label mappings, center crops,
double-resolution output, and morphological post-processing.


## Installation

Python 3.10 or 3.11 and a CUDA-capable TensorFlow environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The MRI data are not included. Extract the BraTS archives before running the
scripts. Each subject directory must contain the four modalities and, for
training, a segmentation:

```text
DATASET_ROOT/
└── subject_id/
    ├── subject_id_t1.nii.gz
    ├── subject_id_t1ce.nii.gz
    ├── subject_id_t2.nii.gz
    ├── subject_id_flair.nii.gz
    └── subject_id_seg.nii.gz
```

Dataset discovery is recursive, so the original `HGG`, `LGG`, and BraTS 2021
nested directory layouts are supported.

## Implementation of the four stages

### 1. Train on BraTS 2021

```bash
python -m scripts.train_brats2021 \
  /path/to/BraTS2021_TrainingData \
  outputs/brats2021
```

The training used 1,151 training subjects, batch size 1, and a
fixed random seed of 1. Pass `--train-count 1151` to reproduce that split when
using the same dataset version and file set.

### 2. Extract NET labels from BraTS 2018

```bash
python -m scripts.extract_brats2018_net \
  /path/to/BraTS2018_TrainingData \
  outputs/brats2021/model_030_0.1000.weights.h5 \
  outputs/brats2018_harmonized \
  --subset-output outputs/brats2018_net_subset \
  --subset-threshold 10000
```

The threshold is the number of NET voxels in the double-resolution prediction.
The script writes both `*_seg_4label_highres.nii.gz` and a downsampled
`*_seg_4label.nii.gz`, plus `net_voxel_counts.csv`.

### 3. Train NET recognition on the BraTS 2018 subset

```bash
python -m scripts.train_brats2018_net \
  outputs/brats2018_net_subset \
  outputs/brats2018_net
```

The script now keeps a validation split by default; set the desired number explicitly with
`--train-count`.

### 4. Add NET labels to BraTS 2021

```bash
python -m scripts.harmonize_brats2021 \
  /path/to/BraTS2021_TrainingData \
  outputs/brats2018_net/model_060_0.1000.weights.h5 \
  outputs/brats2021_harmonized
```

The script uses channel 2 of the four-channel model as its NET prediction,
applies the original binary opening and dilation, and combines the result with
the standard BraTS 2021 ET, TC, and WT regions. It writes double-resolution and
standard-resolution four-compartment masks plus `net_voxel_counts.csv`.

### 5. Train and apply the harmonized model

Both inputs to this stage must contain `*_seg_4label_highres.nii.gz` masks.

```bash
python -m scripts.train_harmonized \
  outputs/brats2018_harmonized \
  outputs/brats2021_harmonized \
  outputs/combined \
  --train-count 1305

python -m scripts.infer \
  /path/to/BraTS_subjects \
  outputs/combined/model_060_0.1000.weights.h5 \
  outputs/predictions
```

Checkpoint names shown above are examples; use the checkpoint actually produced.

## Command-line options

All training scripts support `--epochs`, `--batch-size`, `--seed`,
`--validation-fraction`, `--train-count`, and `--initial-weights`. Run a script
with `--help` for details. Every training run records the exact subject split in
`split.csv` and the learning history in `history.csv`.

## Methodological details

- Modalities are ordered as T1, T2, T1ce, and FLAIR.
- Each modality is center-cropped to `(96, 192, 160)` and independently
  standardized to zero mean and unit variance.
- Tensor layout is channels-first: `(C, Z, Y, X)`.
- PAU-Net outputs masks at twice the input resolution.
- The final network predicts overlapping regions rather than mutually exclusive
  classes. `scripts/infer.py` converts them into the 0–4 compartment encoding.
- NET extraction and final inference use binary opening and dilation, matching
  the morphological clean-up.
- Output origins `(0, -479, 0)` and `(0, -239, 0)` reproduce the original BraTS
  workflow. Verify orientation and spatial metadata before using other datasets.

## Repository layout

```text
brain_tumor_segmentation/
  data.py             BraTS discovery, NIfTI I/O, preprocessing, data sequence
  harmonization.py    cross-dataset NET extraction and mask writing
  labels.py           label-to-region mappings
  model.py            PAU-Net architecture, Dice loss, and metrics
  training.py         shared training workflow
scripts/
  train_brats2021.py
  extract_brats2018_net.py
  train_brats2018_net.py
  harmonize_brats2021.py
  train_harmonized.py
  infer.py
```


