"""Applied ML Suite - Streamlit dashboard for the 23CSE301 capstone.

Tabs: Regression (airfoil noise) | Classification (CTG fetal state) |
Clustering (seismic bumps) | Benchmarks & Model Evaluation.
Run with:  streamlit run app/app.py
"""
from pathlib import Path

import altair as alt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import arff
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Applied ML Suite", page_icon="🧠", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

REG_FEATURES = ["frequency", "log_frequency", "angle_of_attack", "chord_length",
                "free_stream_velocity", "suction_side_displacement_thickness"]
REG_INPUTS = REG_FEATURES[:1] + REG_FEATURES[2:]
CLF_FEATURES = ["LB", "AC", "FM", "UC", "ASTV", "MSTV", "ALTV", "MLTV", "DL", "DS", "DP",
                "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean", "Median",
                "Variance", "Tendency"]
CLF_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathologic"}

# Results copied from the notebooks (README "Results Summary").
CLASSIFICATION_BENCHMARK = pd.DataFrame(
    [
        ("Logistic Regression", "A", 0.8850, 0.8879, 0.8850, 0.8852, 0.9646),
        ("Gaussian Naive Bayes", "A", 0.8146, 0.8685, 0.8146, 0.8310, 0.9288),
        ("Decision Tree", "A", 0.9178, 0.9164, 0.9178, 0.9167, 0.8897),
        ("KNN", "A", 0.8826, 0.8732, 0.8826, 0.8753, 0.9453),
        ("SVC", "A", 0.8967, 0.8917, 0.8967, 0.8924, 0.9633),
        ("Random Forest", "B", 0.9319, 0.9290, 0.9319, 0.9288, 0.9846),
        ("AdaBoost", "B", 0.9108, 0.9070, 0.9108, 0.9080, 0.9488),
        ("Gradient Boosting", "B", 0.9460, 0.9447, 0.9460, 0.9442, 0.9740),
        ("Bagging", "B", 0.9296, 0.9280, 0.9296, 0.9274, 0.9728),
        ("MLP Neural Network", "B", 0.9225, 0.9221, 0.9225, 0.9222, 0.9772),
    ],
    columns=["Model", "Part", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
)
REGRESSION_BENCHMARK = pd.DataFrame(
    [
        ("Linear Regression", 0.5663, 4.6614, 3.6362),
        ("Ridge Regression", 0.5661, 4.6623, 3.6374),
        ("Lasso Regression", 0.5577, 4.7074, 3.6958),
        ("ElasticNet Regression", 0.5504, 4.7462, 3.7405),
        ("Polynomial Regression", 0.7256, 3.7075, 2.8382),
        ("KNN Regression", 0.9112, 2.1092, 1.5182),
        ("SVR (tuned)", 0.9386, 1.7542, 1.2110),
        ("Decision Tree", 0.8899, 2.3485, 1.7395),
        ("Random Forest", 0.9329, 1.8336, 1.3151),
        ("Gradient Boosting", 0.9302, 1.8697, 1.2803),
    ],
    columns=["Model", "R2", "RMSE", "MAE"],
)
CLUSTERING_BENCHMARK = pd.DataFrame(
    [
        ("K-Means", 2, 0.4702, 1.6817, 604.39, "403 / 2,181"),
        ("Agglomerative (Ward)", 2, 0.8320, 0.4258, 577.21, "2,572 / 12"),
    ],
    columns=["Algorithm", "K", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Cluster sizes"],
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .hero {background: linear-gradient(135deg,#141b34,#0e1427); border:1px solid #26304f;
           border-radius:14px; padding:1.4rem 1.8rem; margin-bottom:1rem;}
    .hero h1 {margin:0; font-size:2rem; background:linear-gradient(90deg,#4da3ff,#8a7bff);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
    .hero p {margin:.3rem 0 0; color:#9aa5c4;}
    .kpi {background:#131a2e; border:1px solid #26304f; border-radius:12px;
          padding:.9rem 1rem; text-align:center;}
    .kpi .v {font-size:1.7rem; font-weight:700; color:#4da3ff;}
    .kpi .l {font-size:.72rem; letter-spacing:.06em; color:#9aa5c4; text-transform:uppercase;}
    .result {background:#131a2e; border:1px solid #26304f; border-left:5px solid #ff4b4b;
             border-radius:10px; padding:1rem 1.3rem; margin-top:.8rem;}
    .result .v {font-size:2rem; font-weight:700;}
    .result .l {color:#9aa5c4; font-size:.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- data / models
@st.cache_resource
def load_models():
    return {
        "reg": joblib.load(MODELS_DIR / "regression_model.pkl"),
        "clf": joblib.load(MODELS_DIR / "classification_model.pkl"),
        "clf_scaler": joblib.load(MODELS_DIR / "classification_scaler.pkl"),
        "prep": joblib.load(MODELS_DIR / "clustering_preprocessor.pkl"),
        "kmeans": joblib.load(MODELS_DIR / "kmeans_model.pkl"),
        "agg": joblib.load(MODELS_DIR / "agglomerative_model.pkl"),
        "pca": joblib.load(MODELS_DIR / "clustering_pca.pkl"),
    }


@st.cache_data
def load_airfoil():
    cols = ["frequency", "angle_of_attack", "chord_length", "free_stream_velocity",
            "suction_side_displacement_thickness", "sound_pressure_level"]
    df = pd.read_csv(DATA_DIR / "airfoil_self_noise.dat", sep="\t", header=None, names=cols)
    df["log_frequency"] = np.log1p(df["frequency"])
    return df


@st.cache_data
def load_ctg():
    df = pd.read_excel(DATA_DIR / "CTG.xls", sheet_name="Raw Data").dropna(subset=["NSP"])
    df["NSP"] = df["NSP"].astype(int)
    return df


@st.cache_data
def load_seismic():
    data, _ = arff.loadarff(DATA_DIR / "seismic-bumps.arff")
    df = pd.DataFrame(data)
    for col in df.select_dtypes(include=object):
        df[col] = df[col].str.decode("utf-8")
    return df


@st.cache_data
def regression_test_predictions():
    """Held-out predictions of the saved regressor (same split as the notebook)."""
    m = load_models()
    df = load_airfoil()
    _, test = train_test_split(df, test_size=0.20, random_state=42)
    # The saved Random Forest was fitted on raw (unscaled) features.
    pred = m["reg"].predict(test[REG_FEATURES])
    return test["sound_pressure_level"].to_numpy(), pred


@st.cache_data
def classification_test_accuracy():
    m = load_models()
    df = load_ctg()
    _, test = train_test_split(df, test_size=0.20, random_state=42, stratify=df["NSP"])
    pred = m["clf"].predict(m["clf_scaler"].transform(test[CLF_FEATURES]))
    return accuracy_score(test["NSP"], pred)


@st.cache_data
def clustering_embedding():
    m = load_models()
    df = load_seismic()
    X = m["prep"].transform(df.drop(columns=["class"]))
    return {
        "pca": m["pca"].transform(X),
        "kmeans": m["kmeans"].predict(X),
        "agg": m["agg"].fit_predict(X),
        "X": X,
    }


try:
    models = load_models()
except FileNotFoundError as err:
    st.error(f"Model file missing: {err}. Run the notebooks / scripts to export the models first.")
    st.stop()

y_true_reg, y_pred_reg = regression_test_predictions()
reg_r2 = r2_score(y_true_reg, y_pred_reg)
clf_acc = classification_test_accuracy()
clust = clustering_embedding()


# --------------------------------------------------------------------------- header + KPIs
st.markdown(
    """
    <div class="hero">
      <h1>🧠 Applied ML Suite: Regression, Classification &amp; Clustering</h1>
      <p>Interactive Review 2 dashboard: Airfoil Noise Prediction, Fetal State Classification
         &amp; Seismic Event Segmentation</p>
    </div>
    """,
    unsafe_allow_html=True,
)
kpis = [
    (f"{reg_r2:.3f} R²", f"Regression ({type(models['reg']).__name__})"),
    (f"{clf_acc * 100:.1f}% Acc", f"Classification ({type(models['clf']).__name__})"),
    (f"{models['kmeans'].n_clusters} Clusters", "Seismic segments (K-Means)"),
    (f"{len(clust['X']):,}", "Seismic records clustered"),
]
for col, (value, label) in zip(st.columns(4), kpis):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)
st.write("")

tab_reg, tab_clf, tab_clu, tab_bench = st.tabs([
    "📈 Airfoil Noise (Regression)",
    "🩺 Fetal State (Classification)",
    "🌋 Seismic Segmentation (Clustering)",
    "📊 Benchmarks & Model Evaluation",
])


def style(chart):
    """Readable axis / legend / title text on the dark theme."""
    return (chart.configure_axis(labelColor="#e6e9f2", titleColor="#e6e9f2", gridColor="#26304f")
            .configure_legend(labelColor="#e6e9f2", titleColor="#e6e9f2")
            .configure_title(color="#e6e9f2")
            .configure_view(strokeOpacity=0))


def result_card(label, value):
    st.markdown(f'<div class="result"><div class="l">{label}</div><div class="v">{value}</div></div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- regression tab
with tab_reg:
    st.header("Airfoil Self-Noise Estimator (Regression)")
    st.write(f"Estimate the **sound pressure level (dB)** from aerodynamic parameters using the trained "
             f"**{type(models['reg']).__name__}**.")
    air = load_airfoil()
    left, right = st.columns(2)
    with left:
        st.subheader("Airfoil Geometry")
        chord = st.number_input("Chord length (m)", 0.02, 0.31, 0.1524, 0.001, format="%.4f", key="r_chord")
        thick = st.number_input("Suction side displacement thickness (m)", 0.0004, 0.0585, 0.0050,
                                0.0001, format="%.4f", key="r_thick")
        angle = st.slider("Angle of attack (degrees)", 0.0, 22.2, 5.4, 0.1, key="r_angle")
    with right:
        st.subheader("Flow Conditions")
        freq = st.select_slider("Frequency (Hz)", options=sorted(air["frequency"].unique()),
                                value=1000, key="r_freq")
        velocity = st.select_slider("Free-stream velocity (m/s)", options=[31.7, 39.6, 55.5, 71.3],
                                    value=39.6, key="r_vel")
        st.caption("Ranges match the training data (UCI Airfoil Self-Noise).")

    if st.button("Calculate Sound Pressure Level", type="primary", width="stretch", key="r_go"):
        row = pd.DataFrame([{
            "frequency": freq, "log_frequency": np.log1p(freq), "angle_of_attack": angle,
            "chord_length": chord, "free_stream_velocity": velocity,
            "suction_side_displacement_thickness": thick,
        }])[REG_FEATURES]
        pred = models["reg"].predict(row)[0]
        result_card("Predicted sound pressure level", f"{pred:.2f} dB")
        st.caption(f"Typical error on unseen data: ±{np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)):.2f} dB (test RMSE).")

    with st.expander("About this model"):
        st.write("Features: the five physical inputs plus the engineered `log_frequency = log1p(frequency)`, "
                 "fed to the Random Forest unscaled (tree models are scale-invariant).")

# --------------------------------------------------------------------------- classification tab
with tab_clf:
    st.header("Fetal State Prediction (Classification)")
    st.write(f"Classify a cardiotocography (CTG) exam as Normal, Suspect or Pathologic using the trained "
             f"**{type(models['clf']).__name__}** (final model from the notebook).")
    ctg = load_ctg()
    defaults = ctg[CLF_FEATURES].median()
    int_cols = {"LB", "AC", "FM", "UC", "DL", "DS", "DP", "Width", "Min", "Max", "Nmax", "Nzeros",
                "Mode", "Mean", "Median", "Variance", "Tendency"}
    for f in CLF_FEATURES:
        st.session_state.setdefault(f"c_{f}", float(defaults[f]))

    def load_example():
        choice = st.session_state["c_example"]
        if choice == "Dataset median":
            vals = defaults
        else:
            cls = {v: k for k, v in CLF_LABELS.items()}[choice]
            vals = ctg[ctg["NSP"] == cls][CLF_FEATURES].iloc[0]
        for f in CLF_FEATURES:
            st.session_state[f"c_{f}"] = float(vals[f])

    st.selectbox("Load an example exam", ["Dataset median", "Normal", "Suspect", "Pathologic"],
                 key="c_example", on_change=load_example)

    def num(feature, label):
        lo, hi = float(ctg[feature].min()), float(ctg[feature].max())
        return st.number_input(label, lo, hi, key=f"c_{feature}", help=f"Training range: {lo:g} – {hi:g}")

    left, right = st.columns(2)
    with left:
        st.subheader("Heart Rate & Uterine Activity")
        num("LB", "LB: baseline FHR (bpm)")
        num("AC", "AC: accelerations")
        num("FM", "FM: fetal movements")
        num("UC", "UC: uterine contractions")
        num("DL", "DL: light decelerations")
        num("DS", "DS: severe decelerations")
        num("DP", "DP: prolonged decelerations")
        st.subheader("Variability")
        num("ASTV", "ASTV: abnormal short-term variability (%)")
        num("MSTV", "MSTV: mean short-term variability")
        num("ALTV", "ALTV: abnormal long-term variability (%)")
        num("MLTV", "MLTV: mean long-term variability")
    with right:
        st.subheader("Histogram Features")
        for f, lbl in [("Width", "Width"), ("Min", "Min"), ("Max", "Max"), ("Nmax", "Nmax"),
                       ("Nzeros", "Nzeros"), ("Mode", "Mode"), ("Mean", "Mean"),
                       ("Median", "Median"), ("Variance", "Variance"), ("Tendency", "Tendency (-1, 0, 1)")]:
            num(f, lbl)

    if st.button("Assess Fetal State", type="primary", width="stretch", key="c_go"):
        row = pd.DataFrame([{f: st.session_state[f"c_{f}"] for f in CLF_FEATURES}])[CLF_FEATURES]
        scaled = models["clf_scaler"].transform(row)
        pred = int(models["clf"].predict(scaled)[0])
        proba = models["clf"].predict_proba(scaled)[0]
        result_card("Predicted fetal state", f"{CLF_LABELS[pred]}  ·  {proba.max() * 100:.1f}% confidence")
        prob_df = pd.DataFrame({"State": [CLF_LABELS[c] for c in models["clf"].classes_],
                                "Probability": proba})
        st.altair_chart(
            style(alt.Chart(prob_df).mark_bar().encode(
                x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                y=alt.Y("State:N", sort=None, title=None),
                color=alt.Color("State:N", legend=None,
                                scale=alt.Scale(domain=list(CLF_LABELS.values()),
                                                range=["#2ecc71", "#f1c40f", "#e74c3c"])),
            ).properties(height=130)),
            width="stretch",
        )
        st.caption("Decision-support demo only; not a medical device.")

# --------------------------------------------------------------------------- clustering tab
with tab_clu:
    st.header("Seismic Event Segmentation (Clustering)")
    st.write("Unsupervised grouping of seismic-bump records with **K-Means (k=2)** and "
             "**Agglomerative Hierarchical Clustering (Ward, k=2)**; the hazard `class` label is never used for fitting.")
    seis = load_seismic()
    num_cols = seis.select_dtypes(include=[np.number]).columns.tolist()

    st.subheader("1. Assign a New Event to a Segment")
    left, right = st.columns(2)
    with left:
        seismic_lvl = st.selectbox("Seismic hazard assessment (seismic)", ["a", "b"])
        seismoac = st.selectbox("Seismoacoustic assessment", ["a", "b", "c"])
        shift = st.selectbox("Shift type", ["N", "W"], index=1)
        ghazard = st.selectbox("Hazard from geophones (ghazard)", ["a", "b", "c"])
        genergy = st.number_input("Seismic energy in last shift (genergy)", 100.0, 3_000_000.0,
                                  float(seis["genergy"].median()), 1000.0)
        gpuls = st.number_input("Pulses in last shift (gpuls)", 0.0, 5000.0, float(seis["gpuls"].median()), 10.0)
    with right:
        gdenergy = st.number_input("Energy deviation (gdenergy)", -100.0, 1300.0, float(seis["gdenergy"].median()), 1.0)
        gdpuls = st.number_input("Pulse deviation (gdpuls)", -100.0, 1300.0, float(seis["gdpuls"].median()), 1.0)
        nbumps = st.slider("Seismic bumps in last shift (nbumps)", 0, 10, 0)
        energy = st.number_input("Total bump energy (energy)", 0.0, 500_000.0, 0.0, 1000.0)
        maxenergy = st.number_input("Maximum bump energy (maxenergy)", 0.0, 500_000.0, 0.0, 1000.0)
    algo = st.radio("Algorithm", ["K-Means", "Agglomerative (Ward)"], horizontal=True)

    if st.button("Assign Event Segment", type="primary", width="stretch", key="u_go"):
        new = {c: 0.0 for c in num_cols}
        new.update(genergy=genergy, gpuls=gpuls, gdenergy=gdenergy, gdpuls=gdpuls,
                   nbumps=float(nbumps), energy=energy, maxenergy=maxenergy)
        new.update(seismic=seismic_lvl, seismoacoustic=seismoac, shift=shift, ghazard=ghazard)
        row = pd.DataFrame([new])[models["prep"].feature_names_in_]
        x_new = models["prep"].transform(row)
        if algo == "K-Means":
            seg = int(models["kmeans"].predict(x_new)[0])
        else:
            # sklearn's AgglomerativeClustering has no predict(): assign to the nearest cluster centroid.
            labels = clust["agg"]
            cents = np.vstack([clust["X"][labels == k].mean(axis=0) for k in np.unique(labels)])
            seg = int(np.argmin(((cents - x_new) ** 2).sum(axis=1)))
        st.session_state["u_result"] = (algo, seg, models["pca"].transform(x_new)[0])

    if "u_result" in st.session_state:
        r_algo, r_seg, r_pt = st.session_state["u_result"]
        result_card(f"{r_algo} segment", f"Cluster {r_seg}")

    st.subheader("2. Cluster Structure (PCA projection of all records)")
    view = st.radio("Show clusters from", ["K-Means", "Agglomerative (Ward)"], horizontal=True, key="u_view")
    labels = clust["kmeans"] if view == "K-Means" else clust["agg"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#131a2e")
    sc = ax.scatter(clust["pca"][:, 0], clust["pca"][:, 1], c=labels, cmap="viridis", alpha=0.6, s=14)
    if "u_result" in st.session_state:
        ax.scatter(*st.session_state["u_result"][2][:2], marker="*", s=300, c="#ff4b4b",
                   edgecolors="white", label="Your event")
        ax.legend(facecolor="#131a2e", labelcolor="white")
    ax.set_xlabel("Principal Component 1", color="white")
    ax.set_ylabel("Principal Component 2", color="white")
    ax.tick_params(colors="white")
    ax.set_title(f"{view}: PCA 2D (explained variance {models['pca'].explained_variance_ratio_[:2].sum():.0%})",
                 color="white")
    cbar = fig.colorbar(sc, ax=ax, label="Cluster")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    st.pyplot(fig)
    sizes = pd.Series(labels).value_counts().sort_index()
    st.caption("Cluster sizes: " + ", ".join(f"cluster {k} = {v:,}" for k, v in sizes.items()))
    if view != "K-Means":
        st.warning("Ward k=2 isolates only a handful of outlier events, so its high silhouette score is inflated; "
                   "K-Means gives the more informative segmentation.")

# --------------------------------------------------------------------------- benchmarks tab
with tab_bench:
    st.header("Model Evaluation & Review 2 Benchmarking")

    st.subheader("1. Classification: All 10 Models Comparison")
    bench = CLASSIFICATION_BENCHMARK.copy()
    bench["Model"] = bench["Model"] + " (" + bench["Part"] + ")"
    order = bench.sort_values("F1", ascending=False)["Model"].tolist()
    base = alt.Chart(bench).encode(
        y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False)),
        x=alt.X("F1:Q", scale=alt.Scale(domain=[0.7, 1.0]), title="Weighted F1-score"),
    )
    bars = base.mark_bar().encode(
        color=alt.Color("Part:N", title="Review part", scale=alt.Scale(
            domain=["A", "B"], range=["#4da3ff", "#ff4b4b"])),
        tooltip=["Model", alt.Tooltip("Accuracy:Q", format=".4f"), alt.Tooltip("F1:Q", format=".4f"),
                 alt.Tooltip("ROC-AUC:Q", format=".4f")],
    )
    labels = base.mark_text(align="left", dx=4, color="#e6e9f2").encode(text=alt.Text("F1:Q", format=".4f"))
    names = base.mark_text(align="left", dx=6, color="white", fontWeight="bold").encode(
        x=alt.value(0), text="Model:N")
    st.altair_chart(style((bars + labels + names).properties(height=380, title="10 Classification Models: weighted F1-score")),
                    width="stretch")
    st.dataframe(
        bench.drop(columns="Part").sort_values("F1", ascending=False).reset_index(drop=True).style
        .format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]})
        .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], color="#1f4d2b"),
        width="stretch",
    )
    st.success("Final model: **Gradient Boosting**, highest accuracy and weighted F1. Random Forest has the best ROC-AUC. "
               "Part A = Review 1 models, Part B = Review 2 models.")

    st.divider()
    st.subheader("2. Regression: Actual vs Predicted (held-out test set)")
    pts = pd.DataFrame({"Actual (dB)": y_true_reg, "Predicted (dB)": y_pred_reg})
    lo, hi = float(pts.min().min()), float(pts.max().max())
    scatter = alt.Chart(pts).mark_circle(size=35, opacity=0.6, color="#4da3ff").encode(
        x=alt.X("Actual (dB):Q", scale=alt.Scale(domain=[lo, hi])),
        y=alt.Y("Predicted (dB):Q", scale=alt.Scale(domain=[lo, hi])),
        tooltip=["Actual (dB)", "Predicted (dB)"],
    )
    ideal = alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(color="#ff4b4b", strokeDash=[6, 4]).encode(x="x", y="y")
    st.altair_chart(style((scatter + ideal).properties(
        height=360, title=f"Goodness of fit on test set (R² = {reg_r2:.3f}, "
                          f"MAE = {mean_absolute_error(y_true_reg, y_pred_reg):.2f} dB)")),
        width="stretch")
    st.dataframe(REGRESSION_BENCHMARK.sort_values("R2", ascending=False).reset_index(drop=True).style
                 .format({"R2": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"}), width="stretch")

    st.divider()
    st.subheader("3. Clustering: K-Means vs Agglomerative")
    st.dataframe(CLUSTERING_BENCHMARK, width="stretch", hide_index=True)
    st.caption("Silhouette & Calinski-Harabasz: higher is better. Davies-Bouldin: lower is better.")

st.divider()
st.caption("Applied ML Suite · Regression, Classification & Clustering · Streamlit & scikit-learn · 23CSE301 Review 2")
