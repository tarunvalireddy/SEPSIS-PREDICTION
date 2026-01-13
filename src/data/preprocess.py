import pandas as pd

INPUT_PATH = "data/processed/sepsis_processed.csv"
OUTPUT_PATH = "data/processed/sepsis_final.csv"
PREDICTION_HORIZON = 6

def remove_future_leakage(df):
    final_rows = []

    for pid, pdf in df.groupby("PatientID"):
        sepsis_rows = pdf[pdf["SepsisLabel"] == 1]

        if len(sepsis_rows) > 0:
            onset = sepsis_rows.index[0]
            cutoff = max(onset - PREDICTION_HORIZON, 0)
            pdf = pdf.iloc[:cutoff]
        else:
            # non-sepsis patient → keep full stay
            pdf = pdf.copy()

        final_rows.append(pdf)

    return pd.concat(final_rows, ignore_index=True)

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    df = remove_future_leakage(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Future leakage removed:", df.shape)

