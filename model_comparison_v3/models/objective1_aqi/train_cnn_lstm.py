"""CNN-LSTM on the clean101 daily dataset - GPU-trained (RTX 5060 Laptop
GPU, torch 2.11.0+cu128, verified working with a real matmul before this
build started).

Fixes two issues found in the original models/objective1_aqi/train_cnn_lstm.py
skeleton (see model_comparison_v3/PROJECT_DOCUMENTATION.md):
1. That version computed its OWN independent quantile-based train/test split
   instead of reusing the shared one - not guaranteed to match the other 4
   models' split. This version uses the dataset's existing `split` column
   (fixed date < 2025-01-01 boundary, identical to every other model here).
2. No entry point/driver existed. This version has one.

Architecture unchanged from the original skeleton (1D conv over the feature
vector at each timestep + 2-layer LSTM over the sequence + small head) -
that part was already well-designed, just needed wiring up. Lookback window
redefined in DAYS (SEQ_LEN=14) instead of the skeleton's 8 WEEKS, per this
comparison's daily cadence (see PROJECT_DOCUMENTATION.md section 2.4).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts", "cnn_lstm")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(V3_ROOT, "data_pipeline"))
from build_clean101_daily_dataset import feature_cols  # noqa: E402

sys.path.insert(0, THIS_DIR)
from eval_utils import evaluate, per_station_metrics, timer, check_not_degenerate  # noqa: E402

SEQ_LEN = 14  # days of lookback per training example (daily cadence, not the skeleton's 8 weeks)


class CnnLstmAqi(nn.Module):
    def __init__(self, n_features: int, conv_channels: int = 32, lstm_hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True, num_layers=2, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(lstm_hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def predict_batched(model, X: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], device=device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def build_sequences(df: pd.DataFrame, cols: list[str], target_col: str, seq_len: int = SEQ_LEN):
    X, y, meta = [], [], []
    for station_id, g in df.groupby("station_id"):
        g = g.sort_values("date").reset_index(drop=True)
        feats = g[cols].to_numpy(dtype=np.float32)
        targets = g[target_col].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        for i in range(seq_len, len(g)):
            window = feats[i - seq_len:i]
            if np.isnan(window).any() or np.isnan(targets[i]):
                continue
            X.append(window)
            y.append(targets[i])
            meta.append((station_id, dates[i]))
    return np.array(X), np.array(y), meta


def main(epochs: int = 30, batch_size: int = 64, lr: float = 1e-3):
    df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"), parse_dates=["date"])
    cols = feature_cols(df)
    df = df.sort_values(["station_id", "date"])

    # per-station chronological continuity matters for sequences - a station
    # whose rows are split mid-sequence just yields fewer usable windows at
    # the boundary, no special handling needed beyond using the shared split column
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    scaler = StandardScaler()
    scaler.fit(train_df[cols].fillna(train_df[cols].median()))

    def prep(d):
        d = d.copy()
        d[cols] = scaler.transform(d[cols].fillna(d[cols].median()))
        return build_sequences(d, cols, "aqi")

    X_train, y_train, _ = prep(train_df)
    X_test, y_test, meta_test = prep(test_df)
    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError("Not enough sequential data per station to build CNN-LSTM windows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    model = CnnLstmAqi(n_features=len(cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_train, device=device)
    yt = torch.tensor(y_train, device=device)
    n = len(Xt)

    with timer() as t_train:
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                pred = model(Xt[idx])
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            if (epoch + 1) % 5 == 0:
                print(f"epoch {epoch+1}/{epochs} train_mse={total_loss/n:.2f}")

    model.eval()
    with timer() as t_infer:
        pred_test = predict_batched(model, X_test, device)
    pred_train = predict_batched(model, X_train, device)

    check_not_degenerate(pred_test, "CNN-LSTM")

    metrics = {
        "train": evaluate(y_train, pred_train),
        "test": evaluate(y_test, pred_test),
        "train_seconds": t_train["seconds"],
        "inference_seconds_total": t_infer["seconds"],
        "inference_seconds_per_row": t_infer["seconds"] / len(X_test),
        "n_features": len(cols),
        "seq_len_days": SEQ_LEN,
        "device": device,
    }
    print(json.dumps(metrics, indent=2))

    test_out = pd.DataFrame(meta_test, columns=["station_id", "date"])
    test_out["aqi"] = y_test
    test_out["pred"] = pred_test
    per_station = per_station_metrics(test_out, "aqi", "pred")

    torch.save(model.state_dict(), os.path.join(ARTIFACTS_DIR, "cnn_lstm.pt"))
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    per_station.to_csv(os.path.join(ARTIFACTS_DIR, "per_station_metrics.csv"), index=False)
    test_out.to_csv(os.path.join(ARTIFACTS_DIR, "test_predictions.csv"), index=False)
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "w") as f:
        json.dump(cols, f, indent=2)

    # permutation importance (SHAP's tree explainer doesn't apply to LSTMs) - Phase 2
    base_r2 = evaluate(y_test, pred_test)["r2"]
    importances = []
    rng = np.random.default_rng(42)
    for j, col in enumerate(cols):
        X_perm = X_test.copy()
        rng.shuffle(X_perm[:, :, j].reshape(-1))  # shuffle this feature channel across the batch dim
        pred_perm = predict_batched(model, X_perm, device)
        r2_drop = base_r2 - evaluate(y_test, pred_perm)["r2"]
        importances.append({"feature": col, "r2_drop": r2_drop})
    pd.DataFrame(importances).sort_values("r2_drop", ascending=False).to_csv(
        os.path.join(ARTIFACTS_DIR, "permutation_importance.csv"), index=False)

    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
