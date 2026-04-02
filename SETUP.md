# Setup Guide

## Prerequisites

- Python 3.9+
- Git
- pip

## Environment Setup

```bash
git clone https://github.com/Omarindigo/ForensicsDetective2.git
cd ForensicsDetective2

python -m venv venv
source venv/bin/activate

pip install -r Assignment2/requirements.txt
```

## Verify Data

The starter repository already includes the PNG images generated from PDFs.

- `word_pdfs_png/` contains Word-generated document images
- `google_docs_pdfs_png/` contains Google Docs-generated document images
- `python_pdfs_png/` contains Python/ReportLab-generated document images

Optional submission-aligned layout:

```text
Assignment2/data/original_pdfs/
├── word/
├── google/
└── python/
```

If those `data/original_pdfs/` folders are not populated, the code automatically falls back to the starter repo root folders.

## Running the Pipeline

```bash
cd Assignment2/src
python augmentation.py
python analysis.py
```

## Output

```text
Assignment2/
├── data/augmented_images/
├── results/
│   ├── confusion_matrices/
│   ├── robustness_plots/
│   ├── performance_metrics.csv
│   ├── mcnemar_pvalues.csv
│   └── mcnemar_heatmap.png
```

## Collaborators

- `delveccj`
- `AnushkaTi`
