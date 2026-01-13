import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
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

MAX_PATIENTS = 500              # 🔥 limit for EC2
MAX_WINDOWS_PER_PATIENT = 50    # 🔥 limit per patient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- DATASET ----------------
class SepsisDataset(Dataset):
    def __init__(self, df):
        self.samples = []

        for idx, (pid, pdf) in enumerate(df.groupby("PatientID")):
            if idx >= MAX_PATIENTS:
                break

            if idx % 50 == 0:
                print(f"Building dataset... patient {idx}")

            pdf = pdf.reset_index(drop=True)

            # 🔥 handle missing ICU values
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


# ---------------- MODEL ----------------
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
        x = x[:, -1, :]          # last time step
        return self.fc(x)        # 🔥 raw logits (NO sigmoid)


# ---------------- TRAIN ----------------
def train_transformer():
    df = pd.read_csv(DATA_PATH)

    # patient-wise split (NO leakage)
    patient_ids = df["PatientID"].unique()
    train_pid, test_pid = train_test_split(
        patient_ids, test_size=0.2, random_state=42
    )

    train_df = df[df["PatientID"].isin(train_pid)]
    test_df = df[df["PatientID"].isin(test_pid)]

    train_ds = SepsisDataset(train_df)
    test_ds = SepsisDataset(test_df)

    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples : {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    model = SepsisTransformer(len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()   # 🔥 stable loss

    # -------- TRAIN LOOP --------
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x).squeeze()
            loss = loss_fn(logits, y)
            loss.backward()

            # 🔥 prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss:.4f}")

    # -------- EVALUATION --------
    model.eval()
    probs = []
    labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x).squeeze()
            p = torch.sigmoid(logits).cpu().numpy()

            probs.extend(p)
            labels.extend(y.numpy())

    print("Test label distribution:", Counter(labels))

    if len(set(labels)) > 1:
        print("ROC-AUC :", roc_auc_score(labels, probs))
        print("AUPRC  :", average_precision_score(labels, probs))
    else:
        print("⚠️ Only one class present in test set. ROC-AUC not defined.")
        print("Mean predicted probability:", np.mean(probs))


if __name__ == "__main__":
    train_transformer()

