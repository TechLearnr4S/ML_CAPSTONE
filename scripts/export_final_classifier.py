"""Re-export the final classification model (Gradient Boosting) used by the Streamlit app.

Uses the same CTG preprocessing, 80/20 stratified split and baseline
Gradient Boosting settings as notebooks/classification.ipynb.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES = [
    "LB", "AC", "FM", "UC", "ASTV", "MSTV", "ALTV", "MLTV", "DL", "DS", "DP",
    "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean", "Median",
    "Variance", "Tendency",
]

df = pd.read_excel(BASE_DIR / "data" / "CTG.xls", sheet_name="Raw Data").dropna(subset=["NSP"])
X, y = df[FEATURES], df["NSP"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_train)
# Tree ensembles are scale-invariant; the scaler is kept so the app's
# load/transform/predict flow stays the same for every model.
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(scaler.transform(X_train), y_train)

pred = model.predict(scaler.transform(X_test))
print(f"Accuracy {accuracy_score(y_test, pred):.4f} | weighted F1 {f1_score(y_test, pred, average='weighted'):.4f}")

joblib.dump(model, BASE_DIR / "models" / "classification_model.pkl")
joblib.dump(scaler, BASE_DIR / "models" / "classification_scaler.pkl")
