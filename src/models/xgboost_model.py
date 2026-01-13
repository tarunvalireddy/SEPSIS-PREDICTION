import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/sepsis_final.csv"
TARGET = "EarlySepsisLabel"

def train_xgboost():
    df = pd.read_csv(DATA_PATH)

    # 🚨 CRITICAL: patient-level split
    patient_ids = df["PatientID"].unique()

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=0.2,
        random_state=42
    )

    train_df = df[df["PatientID"].isin(train_ids)]
    test_df = df[df["PatientID"].isin(test_ids)]
    drop_cols = ["SepsisLabel", "EarlySepsisLabel", "PatientID", "Hour"]

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    y_test = test_df[TARGET]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("AUPRC :", average_precision_score(y_test, y_prob))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_xgboost()

