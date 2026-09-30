"""Regenerates the canonical artifacts/cnn_lstm/test_predictions.csv from
the final tuned (smallcap) model, matching the other 4 models' schema
(station_id,date,aqi,pred) - the prior file was from an earlier, since-
superseded tuning iteration (test R2=0.184, now 0.367)."""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
TUNE_DIR = os.path.join(THIS_DIR, "artifacts", "cnn_lstm_tune")
OUT_PATH = os.path.join(THIS_DIR, "artifacts", "cnn_lstm", "test_predictions.csv")

sys.path.insert(0, os.path.join(V3_ROOT, "data_pipeline"))
from build_clean101_daily_dataset import feature_cols  # noqa: E402


class CnnLstmAqi(nn.Module):
    def __init__(self, n_features, conv_channels=16, lstm_hidden=16, num_layers=1, dropout=0.0, head_dropout=0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=1), nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True, num_layers=num_layers,
                             dropout=(dropout if num_layers > 1 else 0.0))
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(lstm_hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_sequences(df, cols, target_col, seq_len):
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


def main():
    with open(os.path.join(TUNE_DIR, "smallcap_meta.json")) as f:
        meta_cfg = json.load(f)
    seq_len, hidden, y_mean, y_std, cols = (meta_cfg["seq_len"], meta_cfg["hidden"], meta_cfg["y_mean"],
                                             meta_cfg["y_std"], meta_cfg["cols"])
    scaler_x = joblib.load(os.path.join(TUNE_DIR, "smallcap_scaler_x.pkl"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CnnLstmAqi(n_features=len(cols), lstm_hidden=hidden, conv_channels=meta_cfg["conv_channels"],
                        num_layers=meta_cfg["num_layers"], dropout=meta_cfg["dropout"],
                        head_dropout=meta_cfg["head_dropout"]).to(device)
    model.load_state_dict(torch.load(os.path.join(TUNE_DIR, "smallcap_model.pt"), map_location=device))
    model.eval()

    df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"), parse_dates=["date"])
    df = df.sort_values(["station_id", "date"])
    test_df = df[df["split"] == "test"].copy()
    test_df[cols] = scaler_x.transform(test_df[cols].fillna(test_df[cols].median()))
    X_test, y_test, meta_test = build_sequences(test_df, cols, "aqi", seq_len)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            xb = torch.tensor(X_test[i:i + 4096], device=device)
            preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds) * y_std + y_mean

    out = pd.DataFrame(meta_test, columns=["station_id", "date"])
    out["aqi"] = y_test
    out["pred"] = pred
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    from sklearn.metrics import r2_score
    print(f"sanity check r2={r2_score(out['aqi'], out['pred']):.4f} (should match metrics.json's 0.367)")


if __name__ == "__main__":
    main()
