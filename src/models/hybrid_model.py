import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import Counter

# ---------------- CONFIG ----------------
DATA_PATH = "data/processed/sepsis_final.csv"
TARGET = "EarlySepsisLabel"

FEATURES = ["HR", "Temp", "Resp", "SBP", "Lactate"]

WINDOW = 6
BATCH_SIZE = 128
EPOCHS = 3
LR = 5e-4

MAX_PATIENTS = 2000
MAX_WINDOWS_PER_PATIENT = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TRANSFORMER DATASET ----------------
class SepsisDataset(Dataset):
    def __init__(self, df):
        self.samples = []

        for idx, (pid, pdf) in enumerate(df.groupby("PatientID")):
            if idx >= MAX_PATIENTS:
                break

            pdf = pdf.reset_index(drop=True)
            pdf[FEATURES] = pdf[FEATURES].ffill().bfill()

            count = 0
            for i in range(WINDOW, len(pdf)):
                if count >= MAX_WINDOWS_PER_PATIENT:
                    break

                x = pdf.iloc[i - WINDOW:i][FEATURES].values
                y = pdf.iloc[i][TARGET]

                if np.isnan(x).any():
                    continue

                self.samples.append((x, y))
                count += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

# ---------------- TRANSFORMER MODEL ----------------
class SepsisTransformer(nn.Module):
    def __init__(self, num_features, d_model=32):
        super().__init__()

        self.embedding = nn.Linear(num_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.fc(x)   # raw logits

# ---------------- HYBRID TRAIN ----------------
def train_hybrid():
    df = pd.read_csv(DATA_PATH)

    # =====================================================
    # 1️⃣ XGBOOST (TABULAR) — KEEP AS IS (WORKS WELL)
    # =====================================================
    drop_cols = ["SepsisLabel", "EarlySepsisLabel", "PatientID", "Hour"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET]

    patient_ids = df["PatientID"].unique()
    train_pid, test_pid = train_test_split(
        patient_ids, test_size=0.2, random_state=42
    )

    train_df = df[df["PatientID"].isin(train_pid)]
    test_df = df[df["PatientID"].isin(test_pid)]

    X_train_tab = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df[TARGET]

    X_test_tab = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    y_test = test_df[TARGET]

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    xgb.fit(X_train_tab, y_train)
    xgb_prob = xgb.predict_proba(X_test_tab)[:, 1]

    # =====================================================
    # 2️⃣ TRANSFORMER (TEMPORAL) — FIXED VERSION
    # =====================================================
    train_ds = SepsisDataset(train_df)
    test_ds = SepsisDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = SepsisTransformer(len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x).squeeze()
            loss = loss_fn(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Transformer Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    model.eval()
    trans_prob = []

    with torch.no_grad():
        for x, _ in test_loader:
            logits = model(x.to(DEVICE)).squeeze()
            p = torch.sigmoid(logits).cpu().numpy()
            trans_prob.extend(p)

    # =====================================================
    # 3️⃣ HYBRID FUSION
    # =====================================================
    min_len = min(len(xgb_prob), len(trans_prob))
    xgb_prob = xgb_prob[:min_len]
    trans_prob = np.array(trans_prob[:min_len])
    y_final = y_test.iloc[:min_len]

    hybrid_prob = 0.6 * xgb_prob + 0.4 * trans_prob

    print("Test label distribution:", Counter(y_final))

    if len(set(y_final)) > 1:
        print("XGBoost AUPRC     :", average_precision_score(y_final, xgb_prob))
        print("Transformer AUPRC :", average_precision_score(y_final, trans_prob))
        print("HYBRID AUPRC      :", average_precision_score(y_final, hybrid_prob))
        print("HYBRID ROC-AUC    :", roc_auc_score(y_final, hybrid_prob))
    else:
        print("⚠️ Only one class present in test set.")
        print("Mean hybrid probability:", np.mean(hybrid_prob))


if __name__ == "__main__":
    train_hybrid()

