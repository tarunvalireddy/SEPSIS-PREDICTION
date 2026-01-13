import numpy as np

# 🔥 SHAP compatibility patch (DO NOT REMOVE)
if not hasattr(np, "int"):
    np.int = int

import os
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
DATA_PATH = "data/processed/sepsis_final.csv"
TARGET = "EarlySepsisLabel"
OUTPUT_DIR = "logs/plots"
TOP_N_FEATURES = 7   # 🔥 change to 5–9 if needed
MAX_SHAP_SAMPLES = 2000   # 🔥 IMPORTANT (speed + stability)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def shap_bar_plot():
    df = pd.read_csv(DATA_PATH)

    drop_cols = ["SepsisLabel", "EarlySepsisLabel", "PatientID", "Hour"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET]

    # ---------------- PATIENT-WISE SPLIT ----------------
    patient_ids = df["PatientID"].unique()
    train_ids, _ = train_test_split(
        patient_ids, test_size=0.2, random_state=42
    )

    train_df = df[df["PatientID"].isin(train_ids)]

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df[TARGET]

    # 🔥 SHAP SUBSET (VERY IMPORTANT FOR REAL DATA)
    if len(X_train) > MAX_SHAP_SAMPLES:
        X_shap = X_train.sample(MAX_SHAP_SAMPLES, random_state=42)
        y_shap = y_train.loc[X_shap.index]
    else:
        X_shap = X_train
        y_shap = y_train

    # ---------------- TRAIN MODEL ----------------
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # ---------------- SHAP ----------------
    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_shap)

    # ---------------- BAR PLOT ----------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
       "Feature": X_shap.columns,
       "Mean |SHAP|": mean_abs_shap
    }).sort_values(by="Mean |SHAP|", ascending=False)

# 🔥 Keep only top N features
    shap_df = shap_df.head(TOP_N_FEATURES).sort_values(by="Mean |SHAP|", ascending=True)



    plt.figure(figsize=(8, 6))
    plt.barh(shap_df["Feature"], shap_df["Mean |SHAP|"])
    plt.xlabel("Mean |SHAP value| (global importance)")
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_bar.png", dpi=300)
    plt.close()

    print("✅ SHAP BAR plot saved to logs/plots/shap_bar.png")


if __name__ == "__main__":
    shap_bar_plot()

