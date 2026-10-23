<div align="center">

# Applied Machine Learning Suite

**Regression · Classification · Clustering — notebooks, saved models and an interactive dashboard**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)

23CSE301 Machine Learning Capstone · B.Tech. CSE, III Year · Academic Year 2026–27

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Repository Structure](#repository-structure)
4. [Quick Start](#quick-start)
5. [Datasets](#datasets)
6. [Methodology](#methodology)
7. [Detailed Results](#detailed-results)
8. [Interactive Dashboard](#interactive-dashboard)
9. [Limitations](#limitations)
10. [Team](#team)

---

## Overview

This project applies a complete machine-learning workflow to three public UCI datasets, one per problem type:

| Track | Dataset | Task | Algorithms | Notebook |
|-------|---------|------|-----------|----------|
| **Regression** | Airfoil Self-Noise | Predict sound pressure level (dB) | 10 | [regression.ipynb](notebooks/regression.ipynb) |
| **Classification** | Cardiotocography (CTG) | Predict fetal state (Normal / Suspect / Pathologic) | 10 | [classification.ipynb](notebooks/classification.ipynb) |
| **Clustering** | Seismic Bumps | Group seismic events without labels | 2 | [clustering.ipynb](notebooks/clustering.ipynb) |

Every track follows the same pipeline: data audit → exploratory analysis → preprocessing → model training → hyperparameter tuning → evaluation → model selection. The best models are exported and served through a Streamlit dashboard.

## Key Results

| Track | Best model | Headline metric |
|-------|-----------|-----------------|
| Regression | Tuned **SVR** | Test R² **0.9386**, RMSE 1.75 dB |
| Classification | **Gradient Boosting** | Accuracy **0.9460**, weighted F1 **0.9442** |
| Clustering | **K-Means** (k = 2) | Silhouette 0.4702, Calinski-Harabasz 604.39 |

- Ensemble methods lead on both supervised tasks; linear models trail on the non-linear airfoil data (R² about 0.56).
- Random Forest has the best classification ROC-AUC (0.9846) and is the runner-up.
- For clustering, Agglomerative (Ward) scores a higher silhouette (0.8320) only because it isolates 12 outlier events; K-Means gives the more informative segmentation. See [Limitations](#limitations).

## Repository Structure

```text
.
├── README.md
├── requirements.txt              pip dependencies
├── environment.yml               conda environment
├── .streamlit/config.toml        dashboard theme
├── data/
│   ├── airfoil_self_noise.dat    regression dataset
│   ├── CTG.xls                   classification dataset
│   └── seismic-bumps.arff        clustering dataset
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb      Part A + Part B + final model selection
│   └── clustering.ipynb          K-Means + Agglomerative
├── models/                       saved .pkl models used by the dashboard
├── app/app.py                    Streamlit dashboard
├── scripts/
│   └── export_final_classifier.py   re-exports the final Gradient Boosting model
└── team_contribution/            individual member notebooks
```

## Quick Start

**Prerequisites:** Python 3.13 (tested; 3.10+ should work) and `pip` or `conda`.

```bash
git clone <your-repository-url>
cd <repository-folder>
```

Install the dependencies with either tool:

```bash
python -m venv venv
venv\Scripts\activate          # macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
```

```bash
conda env create -f environment.yml
conda activate ml-capstone
```

Run the notebooks:

```bash
jupyter notebook
```

Open a notebook from `notebooks/` and choose **Kernel → Restart & Run All**. Run cells in order, since scalers and models are defined before they are reused. `classification.ipynb` takes roughly 20–25 minutes because of the hyperparameter grid searches.

Launch the dashboard:

```bash
streamlit run app/app.py
```

## Datasets

| Dataset | Source | Rows | Features | Target | Notes |
|---------|--------|------|----------|--------|-------|
| **Airfoil Self-Noise** | [UCI](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) | 1,503 | 5 (+1 engineered) | `sound_pressure_level` (dB) | No missing values or duplicates |
| **Cardiotocography** | [UCI](https://archive.ics.uci.edu/dataset/193/cardiotocography) | 2,126 | 21 | `NSP` (3 classes) | 4 rows with missing target dropped; ~78% / 14% / 8% class split |
| **Seismic Bumps** | [UCI](https://archive.ics.uci.edu/dataset/162/seismic+bumps) | 2,584 | 18 (14 numeric, 4 categorical) | `class` (not used for fitting) | 6 duplicate rows kept |

## Methodology

**Regression**
- 80/20 train-test split (`random_state=42`).
- Engineered feature `log_frequency = log1p(frequency)` to reduce right-skew.
- Features standardised for scale-sensitive models; the saved Random Forest uses raw features.
- Tuning with `GridSearchCV` (5-fold CV) for SVR and the tree-based models.

**Classification**
- 80/20 stratified split (`random_state=42`) so class proportions are preserved.
- Part A: Logistic Regression, KNN, Gaussian NB, Decision Tree, SVC. Part B: Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP.
- All 10 models are compared on one table with Accuracy and weighted Precision, Recall and F1, plus One-vs-Rest weighted ROC-AUC.
- The final model is chosen on weighted F1, with ROC-AUC as the tie-breaker, because the classes are imbalanced. The justification is in the notebook's *Final Model Selection* section.

**Clustering**
- `StandardScaler` on numeric features and `OneHotEncoder` on categorical features give 24 features. The ground-truth `class` is excluded and used only for post-hoc interpretation.
- K-Means: k chosen from the elbow curve and silhouette analysis over k = 2–10.
- Agglomerative: Ward, complete and average linkage compared over k = 2–5, with a dendrogram.
- Evaluation: Silhouette, Davies-Bouldin and Calinski-Harabasz. Visualisation: PCA 2D scatter plots for both algorithms and t-SNE plots.

## Detailed Results

### Regression — Airfoil Self-Noise

| # | Model | Test R² | Test RMSE (dB) | Test MAE (dB) |
|---|-------|---------|----------------|---------------|
| 1 | Linear Regression | 0.5663 | 4.6614 | 3.6362 |
| 2 | Ridge Regression | 0.5661 | 4.6623 | 3.6374 |
| 3 | Lasso Regression | 0.5577 | 4.7074 | 3.6958 |
| 4 | ElasticNet Regression | 0.5504 | 4.7462 | 3.7405 |
| 5 | Polynomial Regression | 0.7256 | 3.7075 | 2.8382 |
| 6 | KNN Regression | 0.9112 | 2.1092 | 1.5182 |
| 7 | **SVR (tuned)** | **0.9386** | **1.7542** | **1.2110** |
| 8 | Decision Tree | 0.8899 | 2.3485 | 1.7395 |
| 9 | Random Forest | 0.9329 | 1.8336 | 1.3151 |
| 10 | Gradient Boosting | 0.9302 | 1.8697 | 1.2803 |

The tuned SVR is the best model in the notebook (`C`, `γ` and `epsilon` tuned by 5-fold grid search). The dashboard serves the saved Random Forest (R² 0.9329).

### Classification — Cardiotocography

| # | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|-------|----------|-----------|--------|----|---------|
| 1 | Logistic Regression | 0.8850 | 0.8879 | 0.8850 | 0.8852 | 0.9646 |
| 2 | KNN | 0.8826 | 0.8732 | 0.8826 | 0.8753 | 0.9453 |
| 3 | Gaussian Naive Bayes | 0.8146 | 0.8685 | 0.8146 | 0.8310 | 0.9288 |
| 4 | Decision Tree | 0.9178 | 0.9164 | 0.9178 | 0.9167 | 0.8897 |
| 5 | SVC | 0.8967 | 0.8917 | 0.8967 | 0.8924 | 0.9633 |
| 6 | Random Forest | 0.9319 | 0.9290 | 0.9319 | 0.9288 | 0.9846 |
| 7 | AdaBoost | 0.9108 | 0.9070 | 0.9108 | 0.9080 | 0.9488 |
| 8 | **Gradient Boosting** | **0.9460** | **0.9447** | **0.9460** | **0.9442** | 0.9740 |
| 9 | Bagging | 0.9296 | 0.9280 | 0.9296 | 0.9274 | 0.9728 |
| 10 | MLP | 0.9225 | 0.9221 | 0.9225 | 0.9222 | 0.9772 |

Precision, Recall and F1 are weighted averages; ROC-AUC is One-vs-Rest, weighted.

### Clustering — Seismic Bumps

| Algorithm | K | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Cluster sizes |
|-----------|---|-------------|------------------|---------------------|---------------|
| K-Means | 2 | 0.4702 | 1.6817 | 604.39 | 403 / 2,181 |
| Agglomerative (Ward) | 2 | 0.8320 | 0.4258 | 577.21 | 2,572 / 12 |

## Interactive Dashboard

`app/app.py` is a tabbed Streamlit app that demonstrates all three tracks.

| Tab | What it does |
|-----|--------------|
| **Airfoil Noise (Regression)** | Enter airfoil geometry and flow conditions; the saved Random Forest predicts the sound pressure level in dB. |
| **Fetal State (Classification)** | Enter the 21 CTG measurements or load a Normal / Suspect / Pathologic example; the final Gradient Boosting model returns the class with probabilities. |
| **Seismic Segmentation (Clustering)** | Assign a new seismic event to a K-Means or Agglomerative segment and see it on a PCA projection of all 2,584 records. |
| **Benchmarks & Model Evaluation** | 10-model classification chart and table, regression actual-vs-predicted plot, 10-model regression table and clustering metrics. |

The header KPIs (R², accuracy, number of clusters, records clustered) are computed live from the saved models and bundled datasets. To re-export the final classifier, run `python scripts/export_final_classifier.py`.

> The classification tab is a decision-support demo, not a medical device.

## Limitations

- **Clustering quality:** the seismic-bumps data has weak cluster structure. Agglomerative (Ward, k = 2) separates only 12 outlier events, so its high silhouette score is misleading. Neither algorithm aligns strongly with the hazard `class` label, which is highly imbalanced.
- **Agglomerative in the dashboard:** scikit-learn's `AgglomerativeClustering` has no `predict()`. New events are assigned to the nearest cluster centroid.
- **Single split:** results come from one train-test split (`random_state=42`); no repeated-split confidence intervals are reported.
- **Small datasets:** the classification and clustering sets have a few thousand rows each, so small metric differences between top models should not be over-interpreted.

## Team

A team of three. Contributors, as recorded in the commit history:

| Contributor | Contribution |
|-------------|--------------|
| Shiva Shanmugan | Main repository commits: regression, classification and clustering notebooks |
| Prithesh | Individual contribution notebooks in `team_contribution/` |
| Sivapriyan R | Individual contribution notebooks in `team_contribution/` |

---

<div align="center">

*23CSE301 Machine Learning Capstone · B.Tech. CSE III Year · Academic Year 2026–27*

</div>
