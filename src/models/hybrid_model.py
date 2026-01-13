import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------- CONFIG ----------------
DATA_PATH = "data/processed/sepsis_final.csv"
TARGET = "EarlySepsisLabel"
FEATURES = ["HR", "Temp", "Resp", "SBP", "Lactate"]
WINDOW = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TRANSFORMER ----------------
class SepsisTransformer(nn.Module):
    def __init__(self, num_features, d_model=32):
        super().__init__()
        self.embed = nn.Linear(num_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return torch.sigmoid(self.fc(x))


def build_sequences(df):
    X, y = [], []

    for pid, pdf in df.groupby("PatientID"):
        pdf = pdf.sort_values("Hour")
        for i in range(WINDOW, len(pdf)):
            X.append(pdf.iloc[i-WINDOW:i][FEATURES].values)
            y.append(pdf.iloc[i][TARGET])

    return np.array(X), np.array(y)


# ---------------- HYBRID TRAIN ----------------
def train_hybrid():
    df = pd.read_csv(DATA_PATH)

    # ---------- XGBOOST ----------
    drop_cols = ["SepsisLabel", "EarlySepsisLabel", "PatientID", "Hour"]
    X_tab = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET]

    pid = df["PatientID"].unique()
    train_pid, test_pid = train_test_split(pid, test_size=0.2, random_state=42)

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

    # ---------- TRANSFORMER ----------
    X_seq, y_seq = build_sequences(df)

    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32).to(DEVICE)
    y_train_seq = torch.tensor(y_train_seq, dtype=torch.float32).to(DEVICE)
    X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32).to(DEVICE)

    model = SepsisTransformer(len(FEATURES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    model.train()
    for _ in range(8):
        opt.zero_grad()
        preds = model(X_train_seq).squeeze()
        loss = loss_fn(preds, y_train_seq)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        trans_prob = model(X_test_seq).squeeze().cpu().numpy()

    # ---------- FUSION ----------
    min_len = min(len(xgb_prob), len(trans_prob))
    hybrid_prob = 0.6 * xgb_prob[:min_len] + 0.4 * trans_prob[:min_len]
    y_final = y_test.iloc[:min_len]

    print("XGBoost AUPRC :", average_precision_score(y_final, xgb_prob[:min_len]))
    print("Transformer AUPRC:", average_precision_score(y_final, trans_prob[:min_len]))
    print("HYBRID AUPRC   :", average_precision_score(y_final, hybrid_prob))

    print("HYBRID ROC-AUC :", roc_auc_score(y_final, hybrid_prob))


if __name__ == "__main__":
    train_hybrid()

