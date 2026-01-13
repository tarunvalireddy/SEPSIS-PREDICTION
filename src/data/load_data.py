import os
import glob
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/raw/csv_real/"
PROCESSED_DATA_PATH = "data/processed/"
PREDICTION_HORIZON = 6  # hours before sepsis

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def load_all_patients():
    files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))
    patient_dfs = []

    for file in files:
        df = pd.read_csv(file)
        df["Hour"] = range(len(df))   # ✅ ADD THIS LINE
        df["PatientID"] = os.path.basename(file).replace(".csv", "")
        patient_dfs.append(df)

    return patient_dfs


def create_early_sepsis_label(df):
    df = df.copy()
    df["EarlySepsisLabel"] = 0

    sepsis_indices = df.index[df["SepsisLabel"] == 1].tolist()

    if len(sepsis_indices) > 0:
        onset = sepsis_indices[0]
        start = max(onset - PREDICTION_HORIZON, 0)
        df.loc[start:onset, "EarlySepsisLabel"] = 1

    return df


def clean_patient_df(df):
    # Drop columns with too many missing values
    missing_ratio = df.isna().mean()
    df = df.loc[:, missing_ratio < 0.6]

    # Forward fill (clinical realistic)
    df = df.fillna(method="ffill")

    # Backward fill for early hours
    df = df.fillna(method="bfill")

    return df


def build_dataset():
    patients = load_all_patients()
    processed = []

    for df in patients:
        df = clean_patient_df(df)
        df = create_early_sepsis_label(df)
        processed.append(df)

    full_df = pd.concat(processed, ignore_index=True)
    full_df.to_csv(os.path.join(PROCESSED_DATA_PATH, "sepsis_processed.csv"), index=False)

    print("✅ Dataset created:", full_df.shape)
    print("Early sepsis positive ratio:", full_df["EarlySepsisLabel"].mean())


if __name__ == "__main__":
    build_dataset()

