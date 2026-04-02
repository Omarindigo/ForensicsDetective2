# Assignment 2: ForensicsDetective — Hero or Zero?

EAS 510 — Basics of AI

## Overview

This assignment extends the ForensicsDetective PDF provenance detection project by scaling the dataset through five controlled image augmentations, adding two new classifiers, conducting robustness analysis, and generating a full evaluation pipeline with confusion matrices, summary metrics, and statistical testing.

The experiment design follows the assignment requirement exactly: classifiers are trained only on the original dataset, and augmented datasets are used only for evaluation in order to measure robustness under domain shift.

## File Organization

```text
Assignment2/
├── README.md
├── SETUP.md
├── requirements.txt
├── data/
│   ├── original_pdfs/
│   │   ├── word/
│   │   ├── google/
│   │   └── python/
│   └── augmented_images/
│       ├── gaussian_noise/{word,google,python}/
│       ├── jpeg_compression/{word,google,python}/
│       ├── dpi_downsampling/{word,google,python}/
│       ├── random_cropping/{word,google,python}/
│       └── bit_depth_reduction/{word,google,python}/
├── src/
│   ├── utils.py
│   ├── augmentation.py
│   ├── image_conversion.py
│   ├── classification.py
│   └── analysis.py
├── results/
│   ├── confusion_matrices/
│   ├── robustness_plots/
│   ├── performance_metrics.csv
│   ├── mcnemar_pvalues.csv
│   └── mcnemar_heatmap.png
└── reports/
    └── final_research_report.pdf
```

## Dataset Notes

The original repository already provides PNG images in the starter repo root folders:

- `word_pdfs_png/`
- `google_docs_pdfs_png/`
- `python_pdfs_png/`

This code prefers the submission-aligned `data/original_pdfs/<class>/` folders if they are populated. If they are empty or missing, it automatically falls back to the original starter repo folders so the pipeline still runs cleanly without copying the dataset.

## Required Augmentations

- Gaussian Noise: sigma sampled in `[5, 20]`
- JPEG Compression: quality sampled in `[20, 80]`
- DPI Downsampling: `300 -> 150` or `300 -> 72`
- Random Cropping: `1%–3%` removed from each border
- Bit-Depth Reduction: `8-bit -> 4-bit`

Each augmentation is applied independently to the original image. The pipeline therefore creates exactly five augmented variants per source image.

## Classifiers

The full pipeline evaluates four classifiers:

- SVM with RBF kernel
- SGD classifier with hinge loss
- Random Forest
- Gradient Boosting

This satisfies the requirement to add two additional classifiers beyond the original baseline models.

## How to Run

```bash
cd Assignment2/src
python augmentation.py
python analysis.py
```

If needed, you can also pass the repository root explicitly:

```bash
python augmentation.py /path/to/repo
python analysis.py /path/to/repo
```
