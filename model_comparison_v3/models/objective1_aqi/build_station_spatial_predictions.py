"""Option A spatial predictions: predicted AQI at the 101 real CPCB station
locations, using the tuned CNN-LSTM (early-stopped, test R2=0.339) - not a
full national-grid retrain (see PROJECT_DOCUMENTATION.md / session decision:
CNN-LSTM has no gridded-feature training data, only station-level overpass-
matched satellite features, so a full India grid map is out of scope for now).

For each of the 101 stations, takes that station's most recent SEQ_LEN days
of real feature history and runs it through the trained model to get one
current predicted_aqi per station - directly plottable on the map.
"""

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
OUT_PATH = os.path.join(DATA_DIR, "cnn_lstm_station_spatial_predictions.csv")

sys.path.insert(0, os.path.join(V3_ROOT, "data_pipeline"))
from build_clean101_daily_dataset import feature_cols  # noqa: E402


class CnnLstmAqi(nn.Module):
    def __init__(self, n_features: int, conv_channels: int = 16, lstm_hidden: int = 16,
                 num_layers: int = 1, dropout: float = 0.0, head_dropout: float = 0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True, num_layers=num_layers,
                             dropout=(dropout if num_layers > 1 else 0.0))
        self.head = nn.Sequential(
            nn.Dropout(head_dropout), nn.Linear(lstm_hidden, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def main():
    with open(os.path.join(TUNE_DIR, "smallcap_meta.json")) as f:
        meta = json.load(f)
    seq_len, hidden, y_mean, y_std, cols = meta["seq_len"], meta["hidden"], meta["y_mean"], meta["y_std"], meta["cols"]

    scaler_x = joblib.load(os.path.join(TUNE_DIR, "smallcap_scaler_x.pkl"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CnnLstmAqi(n_features=len(cols), lstm_hidden=hidden, conv_channels=meta["conv_channels"],
                        num_layers=meta["num_layers"], dropout=meta["dropout"],
                        head_dropout=meta["head_dropout"]).to(device)
    model.load_state_dict(torch.load(os.path.join(TUNE_DIR, "smallcap_model.pt"), map_location=device))
    model.eval()

    df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"), parse_dates=["date"])
    assert set(cols).issubset(df.columns), f"missing feature columns: {set(cols) - set(df.columns)}"
    df = df.sort_values(["station_id", "date"])

    rows = []
    skipped = 0
    for station_id, g in df.groupby("station_id"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < seq_len:
            skipped += 1
            continue
        window = g.tail(seq_len).copy()
        window[cols] = scaler_x.transform(window[cols].fillna(window[cols].median()))
        feats = window[cols].to_numpy(dtype=np.float32)
        if np.isnan(feats).any():
            skipped += 1
            continue
        X = torch.tensor(feats[None, :, :], device=device)
        with torch.no_grad():
            pred_s = model(X).cpu().numpy()[0]
        pred = float(pred_s) * y_std + y_mean
        last = g.iloc[-1]
        rows.append({
            "station_id": station_id,
            "city": last.get("city"),
            "lat": float(last["lat"]),
            "lon": float(last["lon"]),
            "date": last["date"].strftime("%Y-%m-%d"),
            "predicted_aqi": round(pred, 1),
            "aqi_category": aqi_category(pred),
        })

    out = pd.DataFrame(rows).sort_values("predicted_aqi", ascending=False)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} station predictions to {OUT_PATH} ({skipped} stations skipped - insufficient/NaN history)")
    print(out["aqi_category"].value_counts())
    print(out["predicted_aqi"].describe())


if __name__ == "__main__":
    main()
