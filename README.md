<div align="center">

# Coreset Selection and Anomaly Detection

### A Linear Algebra Perspective on Industrial Visual Inspection

**Applied Linear Algebra Lab · Ferdowsi University of Mashhad · Spring 2026**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

</div>

---

## Overview

Industrial inspection systems need to identify defective products reliably, but comparing every test image patch against a large reference set can be computationally expensive.

This project explores **coreset selection**: selecting a small, representative subset of feature vectors from a larger dataset while preserving useful geometric information. The selected subset is then used as a reference memory bank for nearest-neighbor anomaly detection.

Using the **bottle category of the MVTec Anomaly Detection dataset**, this notebook connects concepts from applied linear algebra to a practical visual-inspection task. It investigates how leverage scores, singular value decomposition (SVD), and alternative sampling strategies affect anomaly-detection performance.

## Key Topics

- Hat matrix and leverage scores
- Exact and approximate leverage scores using SVD
- Randomized SVD for large feature matrices
- Coreset construction and row sampling
- Greedy K-center selection
- Nearest-neighbor distance for patch-level anomaly scoring
- Pixel-level AUROC evaluation
- Visualization and comparison of sampling strategies

## How It Works

```text
Precomputed image features
          │
          ▼
Build a memory bank from normal training patches
          │
          ▼
Analyze the feature matrix using SVD
          │
          ▼
Select a compact coreset
  ┌──────────────┬──────────────┬─────────────────┐
  │   Leverage   │    Random    │ K-center Greedy │
  │   sampling   │   sampling   │                 │
  └──────────────┴──────────────┴─────────────────┘
          │
          ▼
Measure each test patch's distance to its
nearest feature vector in the selected coreset
          │
          ▼
Evaluate defect/background separation with AUROC
          │
          ▼
Compare performance across coreset sizes
```

## Mathematical Background

### 1. Hat Matrix and Leverage Scores

For a feature matrix \(A \in \mathbb{R}^{n \times d}\), the least-squares projection matrix is

\[
H = A(A^\top A)^{-1}A^\top
\]

when \(A\) has full column rank. Its diagonal entries, \(h_{ii}\), are the leverage scores. They indicate how strongly individual rows contribute to the geometry of the feature space.

### 2. SVD-Based Leverage Scores

Using the thin singular value decomposition,

\[
A = U\Sigma V^\top,
\]

the projection matrix can be written as

\[
H = UU^\top.
\]

Therefore, the leverage score for row \(i\) is

\[
h_{ii} = \sum_j U_{ij}^{2}.
\]

For large matrices, a truncated randomized SVD can estimate leverage scores using a lower-dimensional subspace.

### 3. Coreset Selection Strategies

The notebook compares three strategies:

- **Leverage score sampling:** gives higher sampling probability to rows with higher leverage scores.
- **Random sampling:** selects rows uniformly at random as a baseline.
- **K-center greedy:** iteratively selects points intended to cover the feature space by choosing points far from the current selected set.

### 4. Nearest-Neighbor Anomaly Detection

For a test patch \(q\) and selected coreset \(C\), the anomaly score is the distance to its nearest coreset vector:

\[
s(q) = \min_{c \in C} \|q-c\|_2.
\]

The evaluation uses **pixel-level AUROC** to measure how well these scores distinguish defective patches from normal background patches. The notebook compares detection quality at different coreset sizes.

## Dataset

This project uses precomputed patch features for the **bottle** category of the MVTec Anomaly Detection dataset. The features were extracted using a Wide ResNet-50 backbone, so a GPU or a deep-learning framework is not required for the notebook's feature-space experiments.

The notebook expects a data directory with this structure:

```text
Data/
├── normal/
│   ├── ... .npy
│   └── ...
└── anomaly/
    ├── ..._anomaly.npy
    ├── ..._normal.npy
    └── ...
```

- `normal/` contains feature maps from defect-free training images.
- `anomaly/` contains separate feature arrays for defective regions and normal background patches in test images.
- The memory bank is built by flattening the normal feature maps into patch-level feature vectors.

**Dataset source:** [MVTec AD – Bottle Category (Kaggle)](https://www.kaggle.com/datasets/thtuan/mvtecad-mvtec-anomaly-detection)

The feature arrays are separate from the notebook. Obtain the precomputed feature files from the course-provided source or prepare them according to the notebook's expected format. Do not commit large dataset files to GitHub unless you have checked the dataset's terms and repository-size constraints.

## Getting Started

### Requirements

- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- NumPy
- Matplotlib
- scikit-learn

### 1. Clone the repository

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git clone https://github.com/YOUR_USERNAME/Coreset-Selection-and-Anomaly-Detection.git
cd Coreset-Selection-and-Anomaly-Detection
```

### 2. Create and activate a virtual environment (recommended)

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install numpy matplotlib scikit-learn jupyter
```

### 4. Configure the dataset path

Open `Coreset_Selection_and_Anomaly_Detection.ipynb` and locate the data configuration cell. Set `DATA_DIR` to the location of your local `Data` folder:

```python
DATA_DIR = "/path/to/Data"  # Linux/macOS example
DATA_URL = None
```

On Windows, use a valid path such as `r"C:\path\to\Data"`. Alternatively, configure `DATA_URL` if you have a valid download URL for the expected archive.

### 5. Run the notebook

```bash
jupyter notebook
```

Open `Coreset_Selection_and_Anomaly_Detection.ipynb` and run the cells from top to bottom. The notebook builds the memory bank, analyzes its feature geometry, evaluates coreset strategies, and generates comparison plots.

## Repository Contents

```text
Coreset-Selection-and-Anomaly-Detection/
├── Coreset_Selection_and_Anomaly_Detection.ipynb
├── README.md
└── ...
```

The notebook contains the mathematical explanations, implementation, experiments, visualizations, and written analysis.

## Expected Outputs

When the notebook is run with the required data available, it produces analyses and visualizations including:

- Leverage-score calculations and comparisons with the Hat matrix
- Singular-value spectrum and low-rank feature analysis
- Feature-space and leverage-score visualizations
- Patch-distance distributions for different defect types
- Pixel-level AUROC comparisons across coreset sizes and sampling methods

**Note:** Numerical results depend on the supplied feature files and the execution environment. No performance numbers are claimed here; run the notebook to obtain the results for your dataset.

## Limitations

- The experiments operate on precomputed feature vectors rather than training an image encoder from scratch.
- Results depend on the quality and format of the provided feature arrays.
- Exact SVD and greedy K-center selection can be computationally expensive for large matrices.
- Pixel-level AUROC measures ranking quality; it does not by itself define a deployment-ready anomaly threshold.

## Academic Context

This work was completed as part of the **Applied Linear Algebra** course at **Ferdowsi University of Mashhad** (Spring 2026).

The project demonstrates how linear algebra concepts—especially projection matrices, leverage scores, and SVD—can inform efficient data selection and a practical anomaly-detection pipeline.

## Author

**Saeed Fahimi**  
Computer Engineering · Ferdowsi University of Mashhad

---

<div align="center">

*Exploring the connection between linear algebra, efficient data selection, and industrial anomaly detection.*

</div>
