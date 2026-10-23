# ML Capstone Project — 23CSE301 Machine Learning

> **B.Tech. Computer Science and Engineering — III Year | Academic Year 2026–27**

A complete, end-to-end Machine Learning pipeline built as part of the 23CSE301 capstone project. The project covers three problem tracks — **Regression**, **Classification**, and **Clustering** — each implemented as a clean, fully-run Jupyter Notebook using Python 3, scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Datasets](#datasets)
4. [Results Summary](#results-summary)
5. [Environment Setup](#environment-setup)
6. [How to Run](#how-to-run)
7. [Algorithms Implemented](#algorithms-implemented)
8. [Review Schedule](#review-schedule)
9. [AI Assistance Disclosure](#ai-assistance-disclosure)

---

## Project Overview

| Item | Detail |
|------|--------|
| **Course** | 23CSE301 Machine Learning |
| **Team size** | 3 members |
| **Tracks** | Regression · Classification · Clustering |
| **Primary tools** | Python 3.13, scikit-learn 1.7.2, Pandas 2.3.3 |
| **Submission format** | Jupyter Notebooks + GitHub repository |
| **Total marks** | 50 (25 per review) + up to 2 bonus |

---

## Repository Structure

```
ML_CAPSTONE/
|-- README.md
|-- requirements.txt            <- pip dependencies
|-- environment.yml             <- conda environment
|-- .gitignore
|-- data/
|   |-- airfoil_self_noise.dat  <- Regression dataset (UCI)
|   |-- CTG.xls                 <- Classification dataset (UCI)
|   `-- seismic-bumps.arff      <- Clustering dataset (UCI)
|-- notebooks/
|   |-- regression.ipynb
|   |-- classification.ipynb    <- Part A + Part B + final model selection
|   `-- clustering.ipynb        <- K-Means + Agglomerative
|-- models/                     <- saved .pkl models used by the GUI
|-- app/app.py                  <- Streamlit GUI
`-- team_contribution/          <- individual member notebooks
```

---

## Datasets

### 1. Airfoil Self-Noise — Regression Track

| Property | Value |
|----------|-------|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) |
| **File** | `data/airfoil_self_noise.dat` |
| **Observations** | 1,503 rows |
| **Features** | 5 aerodynamic input features |
| **Target** | `sound_pressure_level` (continuous, dB) |
| **Missing values** | None |
| **Duplicates** | None |

**Feature descriptions:**

| Feature | Description |
|---------|-------------|
| `frequency` | Frequency of the airfoil (Hz) — strongly right-skewed |
| `angle_of_attack` | Angle of attack (degrees) |
| `chord_length` | Chord length (metres) |
| `free_stream_velocity` | Free-stream velocity (m/s) |
| `suction_side_displacement_thickness` | Suction side displacement thickness (metres) |

**Feature engineering:** `log_frequency = log1p(frequency)` — applied to reduce right-skewness before model training, resulting in 6 total input features.

---

### 2. Cardiotocography (CTG) — Classification Track

| Property | Value |
|----------|-------|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/193/cardiotocography) |
| **File** | `data/CTG.xls` (sheet: `Raw Data`) |
| **Observations** | 2,126 (after removing 4 rows with missing target) |
| **Features** | 21 cardiotocographic measurement features |
| **Target** | `NSP` — Fetal State (3 classes) |
| **Missing values** | 4 rows with missing target dropped; remaining handled |
| **Duplicates** | None |

**Target class distribution:**

| Class | Label | Approx. % |
|-------|-------|-----------|
| `1` | Normal | ~78% |
| `2` | Suspect | ~14% |
| `3` | Pathologic | ~8% |

> **Note:** Class imbalance is present. ROC-AUC (One-vs-Rest, weighted) is used as the primary discrimination metric.

### 3. Seismic Bumps - Clustering Track

| Property | Value |
|----------|-------|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/162/seismic+bumps) |
| **File** | `data/seismic-bumps.arff` |
| **Observations** | 2,584 rows, 18 features (14 numeric, 4 categorical) |
| **Ground truth** | `class` (hazardous bump 0/1) - excluded from fitting, used only for post-hoc interpretation |
| **Missing values / duplicates** | 0 / 6 (kept) |
| **Preprocessing** | `StandardScaler` on numeric + `OneHotEncoder` on categorical -> 24 features |

---

## Results Summary

### Regression Track — Airfoil Self-Noise (Review 1)

All 10 models trained on the same 80/20 train-test split (`random_state=42`). Features standardised using `StandardScaler` (fitted on training set only).

| # | Model | Test R² | Test RMSE (dB) | Test MAE (dB) | 5-Fold CV R² |
|---|-------|---------|----------------|---------------|--------------|
| 1 | Linear Regression | 0.5663 | 4.6614 | 3.6362 | — |
| 2 | Ridge Regression | 0.5661 | 4.6623 | 3.6374 | — |
| 3 | Lasso Regression | 0.5577 | 4.7074 | 3.6958 | — |
| 4 | ElasticNet Regression | 0.5504 | 4.7462 | 3.7405 | — |
| 5 | Polynomial Regression | 0.7256 | 3.7075 | 2.8382 | — |
| 6 | KNN Regression | 0.9112 | 2.1092 | 1.5182 | — |
| 7 | **SVR** ⭐ | **0.9386** | **1.7542** | **1.2110** | computed in notebook |
| 8 | Decision Tree | 0.8899 | 2.3485 | 1.7395 | — |
| 9 | Random Forest | 0.9329 | 1.8336 | 1.3151 | computed in notebook |
| 10 | Gradient Boosting | 0.9302 | 1.8697 | 1.2803 | — |

> ⭐ **Best model: Tuned SVR** — highest Test R² (0.9386), lowest RMSE (1.75 dB) and MAE (1.21 dB). GridSearchCV with 5-fold CV used to tune C, γ, and kernel.

---

### Classification Track - CTG Dataset (Review 1 Part A + Review 2 Part B)

All 10 models use the same 80/20 stratified split (`random_state=42`). Precision/Recall/F1 are weighted averages; ROC-AUC is One-vs-Rest weighted. Values come from the consolidated table in `classification.ipynb`.

| # | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|-------|----------|-----------|--------|----|---------|
| 1 | Logistic Regression | 0.8850 | 0.8879 | 0.8850 | 0.8852 | 0.9646 |
| 2 | KNN | 0.8826 | 0.8732 | 0.8826 | 0.8753 | 0.9453 |
| 3 | Gaussian Naive Bayes | 0.8146 | 0.8685 | 0.8146 | 0.8310 | 0.9288 |
| 4 | Decision Tree | 0.9178 | 0.9164 | 0.9178 | 0.9167 | 0.8897 |
| 5 | SVC | 0.8967 | 0.8917 | 0.8967 | 0.8924 | 0.9633 |
| 6 | Random Forest | 0.9319 | 0.9290 | 0.9319 | 0.9288 | 0.9846 |
| 7 | AdaBoost | 0.9108 | 0.9070 | 0.9108 | 0.9080 | 0.9488 |
| 8 | **Gradient Boosting** | 0.9460 | 0.9447 | 0.9460 | 0.9442 | 0.9740 |
| 9 | Bagging | 0.9296 | 0.9280 | 0.9296 | 0.9274 | 0.9728 |
| 10 | MLP | 0.9225 | 0.9221 | 0.9225 | 0.9222 | 0.9772 |

> **Final model: Gradient Boosting** - highest Accuracy and weighted F1. Random Forest has the best ROC-AUC and is the runner-up. Justification and trade-offs are in the "Final Model Selection" section of the notebook.

---

### Clustering Track - Seismic Bumps (Review 2)

| Algorithm | K | Silhouette | Davies-Bouldin | Calinski-Harabasz | Cluster sizes |
|-----------|---|-----------|----------------|-------------------|---------------|
| K-Means | 2 | 0.4702 | 1.6817 | 604.39 | 403 / 2,181 |
| Agglomerative (Ward) | 2 | 0.8320 | 0.4258 | 577.21 | 2,572 / 12 |

> The Agglomerative silhouette is inflated: the split isolates only 12 outlier points. K-Means gives the more informative segmentation. The notebook includes the elbow curve, dendrogram, PCA scatter plots for both algorithms and t-SNE plots.

---

## Environment Setup

### Prerequisites

- Python **3.13** (tested) — Python 3.10+ should work
- `pip` package manager

### Installation

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/<your-username>/ML_CAPSTONE.git
cd ML_CAPSTONE
```

**Step 2 — (Recommended) Create a virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**Step 3 — Install all dependencies:**
```bash
pip install -r requirements.txt
```
Or, with conda:
```bash
conda env create -f environment.yml
conda activate ml-capstone
```

**Key dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | 2.3.3 | Numerical arrays |
| `pandas` | 2.3.3 | Data loading & manipulation |
| `scikit-learn` | 1.7.2 | All ML models, metrics, GridSearchCV |
| `matplotlib` | 3.10.3 | Plots and visualisations |
| `seaborn` | 0.13.2 | Statistical plots and heatmaps |
| `scipy` | 1.16.2 | scikit-learn dependency |
| `xlrd` | 2.0.2 | Reading `.xls` files (CTG dataset) |
| `jupyter` | 1.1.1 | Notebook environment |

---

## How to Run

**Step 1 — Launch Jupyter Notebook:**
```bash
jupyter notebook
```

**Step 2 — Open and run notebooks:**

Navigate to the `notebooks/` folder and open the desired notebook:

| Notebook | Track | Review |
|----------|-------|--------|
| `regression.ipynb` | Regression (Airfoil Self-Noise) | Review 1 |
| `classification.ipynb` | Classification (CTG) — Part A + Part B | Review 1 & 2 |
| `clustering.ipynb` | Clustering | Review 2 |

**Step 3 — Run all cells in order:**

Use **Kernel → Restart & Run All** to execute the notebook top-to-bottom with a fresh kernel. All cells are designed to run without errors in sequence.

> ⚠️ **Important:** Always use *Restart & Run All* before submission. Do not run cells out of order — scalers and model objects must be created before they are referenced downstream.

---

## Algorithms Implemented

### Regression (10 algorithms)
`LinearRegression` · `Ridge` · `Lasso` · `ElasticNet` · `PolynomialRegression` · `KNeighborsRegressor` · `SVR` · `DecisionTreeRegressor` · `RandomForestRegressor` · `GradientBoostingRegressor`

### Classification (10 algorithms)
**Part A:** `LogisticRegression` · `KNeighborsClassifier` · `GaussianNB` · `DecisionTreeClassifier` · `SVC`

**Part B:** `RandomForestClassifier` · `AdaBoostClassifier` · `GradientBoostingClassifier` · `BaggingClassifier` · `MLPClassifier`

### Clustering (2 algorithms)
`KMeans` · `AgglomerativeClustering`

---




*23CSE301 Machine Learning Capstone | B.Tech. CSE III Year | Academic Year 2026–27*

## GUI Application

A tabbed Streamlit dashboard (`app/app.py`) demonstrates all three tracks plus the benchmark results.

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

| Tab | What it does |
|-----|--------------|
| **Airfoil Noise (Regression)** | Enter airfoil geometry and flow conditions; the saved Random Forest predicts the sound pressure level (dB). |
| **Fetal State (Classification)** | Enter the 21 CTG measurements (or load a Normal / Suspect / Pathologic example); the final Gradient Boosting model returns the class with probabilities. |
| **Seismic Segmentation (Clustering)** | Assign a new seismic event to a K-Means / Agglomerative segment and see it on the PCA projection of all 2,584 records. |
| **Benchmarks & Model Evaluation** | 10-model classification comparison (chart + table), regression actual-vs-predicted plot, 10-model regression table and the clustering metrics. |

The header KPIs (R², accuracy, clusters, records) are computed live from the saved models and the bundled datasets. The final classifier can be re-exported with `python scripts/export_final_classifier.py`.
