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
│
├── README.md                   ← This file
├── requirements.txt            ← All Python dependencies with versions
│
├── data/
│   ├── airfoil_self_noise.dat  ← Regression dataset (UCI ML Repository)
│   └── CTG.xls                 ← Classification dataset (UCI ML Repository)
│
├── notebooks/
│   ├── regression.ipynb        ← Regression track (Review 1)
│   ├── classification.ipynb    ← Classification track (Review 1 Part A + Review 2 Part B)
│   └── clustering.ipynb        ← Clustering track (Review 2)  [in progress]
│
└── models/
    └── (saved .pkl model files — optional)
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

### Classification Track — CTG Dataset (Review 1, Part A)

All 5 Part A models trained on the same 80/20 stratified split (`random_state=42`). Features standardised where required (Logistic Regression, KNN, SVC).

| # | Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR, weighted) |
|---|-------|----------|-------------------|----------------|------------|-------------------------|
| 1 | Logistic Regression | 0.8850 | 0.8025 | 0.7603 | 0.7767 | computed in notebook |
| 2 | Gaussian Naive Bayes | — | — | — | — | computed in notebook |
| 3 | Decision Tree | 0.9178 | 0.8429 | 0.8348 | 0.8379 | computed in notebook |
| 4 | KNN | — | — | — | — | computed in notebook |
| 5 | SVC | — | — | — | — | computed in notebook |

> Full metric values (including ROC-AUC) are displayed in the consolidated Part A comparison table within `classification.ipynb` after running all cells.

> **Part B** (Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP) + the 10-algorithm consolidated table will be presented in **Review 2**.

---

### Clustering Track (Review 2 — In Progress)

| Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
|-----------|-----------------|----------------------|------------------------|
| K-Means | TBD | TBD | TBD |
| Agglomerative Hierarchical | TBD | TBD | TBD |

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

### Classification (10 algorithms — Part B pending Review 2)
**Part A:** `LogisticRegression` · `KNeighborsClassifier` · `GaussianNB` · `DecisionTreeClassifier` · `SVC`

**Part B:** `RandomForestClassifier` · `AdaBoostClassifier` · `GradientBoostingClassifier` · `BaggingClassifier` · `MLPClassifier`

### Clustering (2 algorithms)
`KMeans` · `AgglomerativeClustering`

---




*23CSE301 Machine Learning Capstone | B.Tech. CSE III Year | Academic Year 2026–27*

## GUI Application

This project includes a local Streamlit web application that demonstrates the Regression, Classification, and Clustering modules interactively.

To run the application locally:

```bash
streamlit run app/app.py
```

**Available Sections:**
- **Regression**: Predict airfoil sound pressure level based on aerodynamic input features.
- **Classification**: Predict fetal state (Normal, Suspect, Pathologic) from CTG exams.
- **Clustering**: Unsupervised grouping of seismic bump events using K-Means and Agglomerative clustering with PCA visualization.

