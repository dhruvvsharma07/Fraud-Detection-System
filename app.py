import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Credit Card Fraud Detection System")

# -------------------------------
# LOAD DATA (SAFE VERSION)
# -------------------------------
@st.cache_data
def load_data():
    data = fetch_openml("creditcard", version=1, as_frame=True)

    # Features
    X = data.data.copy()

    # Target (THIS IS IMPORTANT)
    y = pd.Series(data.target, name="Class").astype(int)

    # Combine
    df = pd.concat([X, y], axis=1)

    return df

# 🔥 MUST CALL THIS
df = load_data()

# -------------------------------
# BASIC CHECK
# -------------------------------
if "Class" not in df.columns:
    st.error(f"Dataset error. Columns found: {df.columns}")
    st.stop()

# -------------------------------
# SPLIT DATA
# -------------------------------
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# MODEL
# -------------------------------
@st.cache_resource
def train_model():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    return model

model = train_model()

# -------------------------------
# SIDEBAR
# -------------------------------
section = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Analysis"]
)

# -------------------------------
# PREDICTION
# -------------------------------
if section == "Prediction":

    st.header("🔮 Fraud Prediction")

    idx = st.slider("Select Transaction", 0, len(X_test)-1, 0)
    sample = X_test.iloc[idx]

    st.write("### Transaction Data")
    st.dataframe(sample)

    if st.button("Predict"):
        pred = model.predict(sample.values.reshape(1, -1))[0]
        prob = model.predict_proba(sample.values.reshape(1, -1))[0][1]

        if pred == 1:
            st.error(f"🚨 Fraud Detected (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ Legit Transaction (Confidence: {1 - prob:.2f})")

# -------------------------------
# ANALYSIS
# -------------------------------
else:

    st.header("📊 Dataset Analysis")

    st.write("Shape:", df.shape)

    st.subheader("Class Distribution")
    st.bar_chart(df["Class"].value_counts())

    st.subheader("Statistics")
    st.dataframe(df.describe())

    # ---------------- HISTOGRAMS ----------------
    st.subheader("Feature Distributions")

    cols = df.columns[:12]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))

    for i, col in enumerate(cols):
        ax = axes[i // 4, i % 4]
        df[col].hist(ax=ax)
        ax.set_title(col)

    plt.tight_layout()
    st.pyplot(fig)

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("Confusion Matrix")

    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax_cm)
    st.pyplot(fig_cm)

    # ---------------- ROC ----------------
    st.subheader("ROC Curve")

    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
    st.pyplot(fig_roc)

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("Feature Importance")

    importances = model.named_steps["clf"].feature_importances_

    fig_imp, ax_imp = plt.subplots()
    ax_imp.bar(range(len(importances)), importances)
    ax_imp.set_title("Feature Importance")
    st.pyplot(fig_imp)