"""CNN-LSTM + learned per-station embedding. Rationale: the LSTM currently
sees station identity only via one scalar (station_aqi_target_enc) diluted
among 65 features at every timestep - a tree model (XGBoost) can split on
that scalar far more effectively than an LSTM can extract it through a
conv+recurrent stack. A learned embedding gives the model an explicit,
higher-capacity per-station identity signal, injected once into the final
representation (not repeated every timestep, since it's static per station).

Reuses the same target-scaling + early-stopping approach that fixed the
overfitting problem in tune_cnn_lstm.py (val R2 there peaked at epoch 1)."""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
TUNE_DIR = os.path.join(THIS_DIR, "artifacts", "cnn_lstm_tune")
os.makedirs(TUNE_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(V3_ROOT, "data_pipeline"))
from build_clean101_daily_dataset import feature_cols  # noqa: E402

sys.path.insert(0, THIS_DIR)
from eval_utils import evaluate, timer, check_not_degenerate  # noqa: E402

SEQ_LEN = 14


class CnnLstmStationAqi(nn.Module):
    def __init__(self, n_features: int, n_stations: int, embed_dim: int = 8, conv_channels: int = 32, lstm_hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=1), nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True, num_layers=2, dropout=0.2)
        self.station_embed = nn.Embedding(n_stations, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + embed_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x, station_idx):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        emb = self.station_embed(station_idx)
        combined = torch.cat([out[:, -1, :], emb], dim=1)
        return self.head(combined).squeeze(-1)


def predict_batched(model, X, sidx, device, batch_size=4096):
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], device=device)
            sb = torch.tensor(sidx[i:i + batch_size], device=device, dtype=torch.long)
            preds.append(model(xb, sb).cpu().numpy())
    return np.concatenate(preds)


def build_sequences(df, cols, target_col, station_to_idx, seq_len=SEQ_LEN):
    X, y, sidx, meta = [], [], [], []
    for station_id, g in df.groupby("station_id"):
        g = g.sort_values("date").reset_index(drop=True)
        feats = g[cols].to_numpy(dtype=np.float32)
        targets = g[target_col].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        idx = station_to_idx[station_id]
        for i in range(seq_len, len(g)):
            window = feats[i - seq_len:i]
            if np.isnan(window).any() or np.isnan(targets[i]):
                continue
            X.append(window)
            y.append(targets[i])
            sidx.append(idx)
            meta.append((station_id, dates[i]))
    return np.array(X), np.array(y), np.array(sidx, dtype=np.int64), meta


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"), parse_dates=["date"])
    cols = feature_cols(df)
    df = df.sort_values(["station_id", "date"])
    station_to_idx = {sid: i for i, sid in enumerate(sorted(df["station_id"].unique()))}
    n_stations = len(station_to_idx)

    train_df_full = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    cutoff = train_df_full["date"].quantile(0.85)
    fit_df = train_df_full[train_df_full["date"] < cutoff]
    val_df = train_df_full[train_df_full["date"] >= cutoff]

    scaler_x = StandardScaler()
    scaler_x.fit(fit_df[cols].fillna(fit_df[cols].median()))

    def prep(d):
        d = d.copy()
        d[cols] = scaler_x.transform(d[cols].fillna(d[cols].median()))
        return build_sequences(d, cols, "aqi", station_to_idx)

    X_fit, y_fit, s_fit, _ = prep(fit_df)
    X_val, y_val, s_val, _ = prep(val_df)
    X_test, y_test, s_test, _ = prep(test_df)

    y_mean, y_std = float(y_fit.mean()), float(y_fit.std())
    y_fit_s = (y_fit - y_mean) / y_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} n_stations={n_stations} n_fit={len(X_fit)} n_val={len(X_val)} n_test={len(X_test)}")
    model = CnnLstmStationAqi(n_features=len(cols), n_stations=n_stations).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_fit, device=device)
    yt = torch.tensor(y_fit_s, device=device)
    St = torch.tensor(s_fit, device=device, dtype=torch.long)
    n = len(Xt)
    batch_size = 64

    best_val_r2 = -1e9
    best_state = None
    epochs_since_improve = 0
    patience = 4
    max_epochs = 30
    val_curve = []

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx], St[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        model.eval()
        val_pred = predict_batched(model, X_val, s_val, device) * y_std + y_mean
        val_r2 = evaluate(y_val, val_pred)["r2"]
        val_curve.append(val_r2)
        print(f"epoch {epoch+1}/{max_epochs} train_mse={total_loss/n:.4f} val_r2={val_r2:.4f} best={best_val_r2:.4f}")
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if epochs_since_improve >= patience:
            print(f"early stop at epoch {epoch+1}, best_val_r2={best_val_r2:.4f}")
            break

    model.load_state_dict(best_state)
    model.eval()
    pred_test = predict_batched(model, X_test, s_test, device) * y_std + y_mean
    check_not_degenerate(pred_test, "CNN-LSTM-station")
    test_metrics = evaluate(y_test, pred_test)
    print(f"\nTRUE TEST: r2={test_metrics['r2']:.4f} rmse={test_metrics['rmse']:.2f} mae={test_metrics['mae']:.2f} n={test_metrics['n']}")

    result = {"test": test_metrics, "val_curve": val_curve, "best_val_r2": best_val_r2, "n_stations": n_stations}
    with open(os.path.join(TUNE_DIR, "station_embed_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), os.path.join(TUNE_DIR, "station_embed_model.pt"))
    joblib.dump(scaler_x, os.path.join(TUNE_DIR, "station_embed_scaler_x.pkl"))
    with open(os.path.join(TUNE_DIR, "station_embed_meta.json"), "w") as f:
        json.dump({"seq_len": SEQ_LEN, "y_mean": y_mean, "y_std": y_std, "cols": cols,
                    "station_to_idx": station_to_idx}, f, indent=2)
    print(f"Saved to {TUNE_DIR}")


if __name__ == "__main__":
    main()
