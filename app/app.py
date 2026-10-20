import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Capstone Project", layout="wide")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

@st.cache_resource
def load_regression_model():
    model = joblib.load(MODELS_DIR / "regression_model.pkl")
    scaler = joblib.load(MODELS_DIR / "regression_scaler.pkl")
    return model, scaler

@st.cache_resource
def load_classification_model():
    model = joblib.load(MODELS_DIR / "classification_model.pkl")
    scaler = joblib.load(MODELS_DIR / "classification_scaler.pkl")
    return model, scaler

@st.cache_resource
def load_clustering_models():
    preprocessor = joblib.load(MODELS_DIR / "clustering_preprocessor.pkl")
    kmeans = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    agg = joblib.load(MODELS_DIR / "agglomerative_model.pkl")
    pca = joblib.load(MODELS_DIR / "clustering_pca.pkl")
    return preprocessor, kmeans, agg, pca

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Regression", "Classification", "Clustering", "About Project"])

if page == "Home":
    st.title("Machine Learning Capstone Project")
    st.subheader("Regression • Classification • Clustering")
    st.write("Welcome to the interactive demonstration of the Machine Learning Capstone Project.")
    st.write("Select a module from the sidebar to begin.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Regression", "10 algorithms")
    with col2:
        st.metric("Classification", "10 algorithms")
    with col3:
        st.metric("Clustering", "2 algorithms")
        
elif page == "Regression":
    st.title("Regression: Airfoil Self-Noise")
    st.write("Predict the sound pressure level (dB) based on airfoil aerodynamics.")
    
    try:
        reg_model, reg_scaler = load_regression_model()
    except FileNotFoundError:
        st.error("Regression model files not found. Please export models first.")
        st.stop()
        
    st.header("Input Features")
    col1, col2 = st.columns(2)
    with col1:
        frequency = st.number_input("Frequency (Hz)", min_value=1.0, value=800.0)
        angle_of_attack = st.number_input("Angle of Attack (degrees)", min_value=-10.0, max_value=30.0, value=0.0)
        chord_length = st.number_input("Chord Length (m)", min_value=0.0, value=0.3048)
    with col2:
        free_stream_velocity = st.number_input("Free-Stream Velocity (m/s)", min_value=0.0, value=71.3)
        suction_side_displacement_thickness = st.number_input("Suction Side Displacement Thickness (m)", min_value=0.0, value=0.00266)
    
    if st.button("Predict Sound Pressure Level"):
        log_frequency = np.log(frequency)
        input_data = pd.DataFrame([{
            "frequency": frequency,
            "log_frequency": log_frequency,
            "angle_of_attack": angle_of_attack,
            "chord_length": chord_length,
            "free_stream_velocity": free_stream_velocity,
            "suction_side_displacement_thickness": suction_side_displacement_thickness
        }])
        
        try:
            input_scaled = reg_scaler.transform(input_data)
            pred = reg_model.predict(input_scaled)[0]
            st.metric("Predicted Sound Pressure Level", f"{pred:.2f} dB")
        except Exception as e:
            st.error(f"Unable to generate prediction. Please verify the input values. Error: {str(e)}")
            
    with st.expander("About this model"):
        st.write("The active model is a **Random Forest Regressor** or **SVR** (as determined by the final notebook export).")
        st.write("Features include the engineered `log_frequency` which is computed automatically before passing to the model.")
        
elif page == "Classification":
    st.title("Classification: Fetal Cardiotocography")
    st.write("Predict the fetal state (Normal, Suspect, Pathologic) based on CTG exams.")
    
    try:
        clf_model, clf_scaler = load_classification_model()
    except FileNotFoundError:
        st.error("Classification model files not found. Please export models first.")
        st.stop()
        
    st.header("Input Features")
    
    with st.expander("Fetal Heart Rate & Uterine Activity"):
        col1, col2, col3 = st.columns(3)
        with col1:
            lb = st.number_input("LB (Baseline FHR)", value=120.0)
            ac = st.number_input("AC (Accelerations)", value=0.0)
            fm = st.number_input("FM (Fetal Movement)", value=0.0)
        with col2:
            uc = st.number_input("UC (Uterine Contractions)", value=0.0)
            dl = st.number_input("DL (Light Decelerations)", value=0.0)
            ds = st.number_input("DS (Severe Decelerations)", value=0.0)
        with col3:
            dp = st.number_input("DP (Prolongued Decelerations)", value=0.0)

    with st.expander("Variability Features"):
        col1, col2 = st.columns(2)
        with col1:
            astv = st.number_input("ASTV (Abnormal Short Term Variability %)", value=50.0)
            altv = st.number_input("ALTV (Abnormal Long Term Variability %)", value=10.0)
        with col2:
            mstv = st.number_input("MSTV (Mean Short Term Variability)", value=1.0)
            mltv = st.number_input("MLTV (Mean Long Term Variability)", value=5.0)

    with st.expander("Histogram Features"):
        col1, col2, col3 = st.columns(3)
        with col1:
            width = st.number_input("Width", value=50.0)
            min_v = st.number_input("Min", value=100.0)
            max_v = st.number_input("Max", value=150.0)
        with col2:
            nmax = st.number_input("Nmax", value=2.0)
            nzeros = st.number_input("Nzeros", value=0.0)
            mode = st.number_input("Mode", value=130.0)
        with col3:
            mean = st.number_input("Mean", value=130.0)
            median = st.number_input("Median", value=130.0)
            variance = st.number_input("Variance", value=10.0)
            tendency = st.number_input("Tendency (-1, 0, 1)", value=0.0)
            
    if st.button("Predict Fetal State"):
        input_data = pd.DataFrame([{
            "LB": lb, "AC": ac, "FM": fm, "UC": uc,
            "ASTV": astv, "MSTV": mstv, "ALTV": altv, "MLTV": mltv,
            "DL": dl, "DS": ds, "DP": dp, "Width": width,
            "Min": min_v, "Max": max_v, "Nmax": nmax, "Nzeros": nzeros,
            "Mode": mode, "Mean": mean, "Median": median, "Variance": variance,
            "Tendency": tendency
        }])
        
        try:
            input_scaled = clf_scaler.transform(input_data)
            pred = clf_model.predict(input_scaled)[0]
            
            mapping = {1: "NORMAL", 2: "SUSPECT", 3: "PATHOLOGIC"}
            st.metric("Predicted Condition", mapping.get(pred, "UNKNOWN"))
            
            if hasattr(clf_model, "predict_proba"):
                probas = clf_model.predict_proba(input_scaled)[0]
                classes = clf_model.classes_
                st.write("**Probabilities:**")
                for c, p in zip(classes, probas):
                    st.write(f"{mapping.get(c, 'UNKNOWN')}: {p*100:.2f}%")
        except Exception as e:
            st.error(f"Unable to generate prediction. Please verify the input values. Error: {str(e)}")

    with st.expander("Model Evaluation — Test Set"):
        st.write("This model was evaluated over the full classification test set with Weighted F1 scoring.")
        
elif page == "Clustering":
    st.title("Clustering: Seismic Bumps")
    st.write("Unsupervised grouping of seismic bump events using K-Means and Agglomerative clustering.")
    
    try:
        prep, kmeans_model, agg_model, pca_model = load_clustering_models()
    except FileNotFoundError:
        st.error("Clustering model files not found. Please export models first.")
        st.stop()
        
    st.header("Clustering Demonstration")
    uploaded_file = st.file_uploader("Upload Seismic Bumps dataset (.arff)", type=["arff"])
    
    if uploaded_file is not None:
        try:
            from scipy.io import arff
            import io
            
            # Read ARFF from uploaded file properly
            data, meta = arff.loadarff(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            df = pd.DataFrame(data)
            
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            
            st.success("Dataset uploaded successfully!")
            
            # Use original preprocessing from notebook (do not use ground-truth class)
            if "class" in df.columns:
                X = df.drop(columns=["class"])
            else:
                X = df
                
            X_processed = prep.transform(X)
            
            algo = st.radio("Select Algorithm", ["K-Means", "Agglomerative Hierarchical Clustering"])
            
            if algo == "K-Means":
                labels = kmeans_model.predict(X_processed)
                n_clusters = kmeans_model.n_clusters
            else:
                # Agglomerative doesn't natively support predict() for new instances using sklearn's implementation
                # If we're uploading the same dataset, we just fit_predict for demonstration.
                labels = agg_model.fit_predict(X_processed)
                n_clusters = agg_model.n_clusters
                
            st.write(f"**Number of Clusters:** {n_clusters}")
            
            st.write("**Cluster Distribution:**")
            counts = pd.Series(labels).value_counts().sort_index()
            for cluster_id, count in counts.items():
                st.write(f"Cluster {cluster_id} → {count} observations")
                
            # Visualization
            st.subheader("PCA Visualization")
            X_pca = pca_model.transform(X_processed)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title(f"{algo} Clusters — PCA 2D Visualization")
            plt.colorbar(scatter, ax=ax, label="Cluster")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error processing dataset: {e}")

elif page == "About Project":
    st.title("About Project")
    
    st.subheader("Project Tracks")
    st.write("- **Regression**")
    st.write("- **Classification**")
    st.write("- **Clustering**")
    
    st.subheader("Technologies")
    st.write("Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit")
    
    st.subheader("Datasets")
    st.write("- **Airfoil Self-Noise** (Regression)")
    st.write("- **Cardiotocography** (Classification)")
    st.write("- **Seismic Bumps** (Clustering)")
    
    st.subheader("Algorithms")
    st.write("**Regression:** 10 algorithms (Linear, Ridge, Lasso, ElasticNet, Polynomial, KNN, SVR, Decision Tree, Random Forest, Gradient Boosting)")
    st.write("**Classification:** 10 algorithms (Logistic, KNN, Gaussian NB, Decision Tree, SVC, Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP)")
    st.write("**Clustering:** K-Means, Agglomerative Hierarchical Clustering")
    
    st.subheader("Results Dashboard (Test Metrics Summary)")
    st.write("**Regression (Top Models)**")
    st.table(pd.DataFrame({
        "Algorithm": ["SVR", "Random Forest"],
        "R²": [0.9386, 0.9329],
        "RMSE": [1.7542, 1.8336],
        "MAE": [1.2110, 1.3151]
    }))
    
    st.write("**Classification (Sample of Top Models)**")
    st.table(pd.DataFrame({
        "Algorithm": ["Random Forest", "Gradient Boosting"],
        "Accuracy": [0.9413, 0.9507],
        "F1 (Weighted)": [0.9398, 0.9497],
        "ROC-AUC": [0.9795, 0.9859]
    }))
    
    st.write("**Clustering**")
    st.table(pd.DataFrame({
        "Algorithm": ["K-Means", "Agglomerative"],
        "K": [2, 2],
        "Silhouette": [0.4702, 0.8320],
        "Davies-Bouldin": [1.6817, 0.4258],
        "Calinski-Harabasz": [604.39, 577.21]
    }))
