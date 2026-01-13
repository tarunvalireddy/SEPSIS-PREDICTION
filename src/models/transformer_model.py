import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

DATA_PATH = "data/processed/sepsis_final.csv"
TARGET = "EarlySepsisLabel"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW = 6

FEATURES = ["HR", "Temp", "Resp", "SBP", "Lactate"]

class SepsisTransformer(nn.Module):
    def __init__(self, num_features, d_model=32, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]      # last time step
        return torch.sigmoid(self.fc(x))


def build_sequences(df):
    X, y = [], []

    for pid, pdf in df.groupby("PatientID"):
        if "Hour" in pdf.columns:
            pdf = pdf.sort_values("Hour")
        else:
            pdf = pdf.reset_index(drop=True)


        for i in range(WINDOW, len(pdf)):
            seq = pdf.iloc[i-WINDOW:i][FEATURES].values
            X.append(seq)
            y.append(pdf.iloc[i][TARGET])

    return np.array(X), np.array(y)


def train_transformer():
    df = pd.read_csv(DATA_PATH)

    X, y = build_sequences(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    model = SepsisTransformer(len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        preds = model(X_train).squeeze()
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        probs = model(X_test).squeeze().cpu().numpy()

    print("ROC-AUC:", roc_auc_score(y_test.cpu(), probs))
    print("AUPRC :", average_precision_score(y_test.cpu(), probs))


if __name__ == "__main__":
    train_transformer()

