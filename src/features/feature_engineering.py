import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/sepsis_processed.csv"
OUTPUT_PATH = "data/processed/sepsis_features.csv"

WINDOW = 6  # hours

VITAL_COLS = [
    "HR", "SBP", "DBP", "Temp", "Resp",
    "SpO2", "WBC", "Lactate", "Creatinine", "Platelets"
]

def add_temporal_features(df):
    df = df.copy()

    for col in VITAL_COLS:
        if col not in df.columns:
            continue

        df[f"{col}_mean"] = (
            df.groupby("PatientID")[col]
            .rolling(WINDOW, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        df[f"{col}_std"] = (
            df.groupby("PatientID")[col]
            .rolling(WINDOW, min_periods=1)
            .std()
            .reset_index(0, drop=True)
            .fillna(0)
        )

        df[f"{col}_delta"] = (
            df.groupby("PatientID")[col]
            .diff()
            .fillna(0)
        )

        df[f"{col}_min"] = (
            df.groupby("PatientID")[col]
            .rolling(WINDOW, min_periods=1)
            .min()
            .reset_index(0, drop=True)
        )

        df[f"{col}_max"] = (
            df.groupby("PatientID")[col]
            .rolling(WINDOW, min_periods=1)
            .max()
            .reset_index(0, drop=True)
        )

    return df


def build_feature_dataset():
    df = pd.read_csv(INPUT_PATH)
    df = add_temporal_features(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Feature dataset created:", df.shape)


if __name__ == "__main__":
    build_feature_dataset()

