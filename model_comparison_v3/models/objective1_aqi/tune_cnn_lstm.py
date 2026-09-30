"""Systematic tuning pass for CNN-LSTM, requested by user (2026-08-07,
time-boxed to ~60 min). Diagnostic findings before this script was written:

1. Feature parity: CONFIRMED identical to XGBoost - both call the same
   `feature_cols()` from build_clean101_daily_dataset.py (65 features:
   satellite AOD/gases, ERA5 weather, calendar, lag1/3/7 + roll3/7 of
   those vars, plus station_aqi_target_enc). No gap.
2. Input scaling: X was already StandardScaler'd in train_cnn_lstm.py.
   TARGET y was NOT scaled (raw AQI, 0-500ish range) - a known high-impact
   gap for neural nets trained with MSE+Adam (trees don't need this, which
   is why XGBoost never had it). Fixed here: target scaling on train
   only, inverse-transformed before computing metrics (so reported
   RMSE/MAE/R2 stay in real AQI units, comparable to the other 4 models).
3. Convergence: the original 30-epoch run's printed loss
   (3860->3193->2906->2733->2623->2537 at epochs 5/10/15/20/25/30) was
   STILL decreasing at a meaningful rate at epoch 30 - i.e. genuinely
   undertrained, not converged. This script trains longer with an LR
   scheduler and logs the full per-epoch curve to check real convergence.

Uses a chronological validation split carved out of the TRAIN period only
(last 15% of train dates) for the sequence-length sweep, so the true test
set (2025) is touched exactly once, at the end, for the final reported
number - avoids quietly overfitting hyperparameter choices to the test set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

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


class CnnLstmAqi(nn.Module):
    def __init__(self, n_features: int, conv_channels: int = 32, lstm_hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.2, head_dropout: float = 0.0):
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


def predict_batched(model, X: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], device=device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def build_sequences(df: pd.DataFrame, cols: list[str], target_col: str, seq_len: int):
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


def load_base():
    df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"), parse_dates=["date"])
    cols = feature_cols(df)
    df = df.sort_values(["station_id", "date"])
    return df, cols


def run_config(df, cols, seq_len, epochs, lr, hidden, use_val_split, batch_size=64, tag="",
               weight_decay=0.0, early_stop=False, patience=4, final_test_check=False,
               num_layers=2, dropout=0.2, head_dropout=0.0, conv_channels=32):
    train_df_full = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    if use_val_split or early_stop:
        cutoff = train_df_full["date"].quantile(0.85)
        fit_df = train_df_full[train_df_full["date"] < cutoff]
        eval_df = train_df_full[train_df_full["date"] >= cutoff]
    else:
        fit_df = train_df_full
        eval_df = test_df

    scaler_x = StandardScaler()
    scaler_x.fit(fit_df[cols].fillna(fit_df[cols].median()))

    def prep_x(d):
        d = d.copy()
        d[cols] = scaler_x.transform(d[cols].fillna(d[cols].median()))
        return build_sequences(d, cols, "aqi", seq_len)

    X_fit, y_fit, _ = prep_x(fit_df)
    X_eval, y_eval, meta_eval = prep_x(eval_df)
    if len(X_fit) == 0 or len(X_eval) == 0:
        raise RuntimeError(f"seq_len={seq_len}: not enough rows to build sequences")

    # target scaling - fit on fit_df targets only
    y_mean, y_std = float(y_fit.mean()), float(y_fit.std())
    y_fit_s = (y_fit - y_mean) / y_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CnnLstmAqi(n_features=len(cols), lstm_hidden=hidden, num_layers=num_layers,
                        dropout=dropout, head_dropout=head_dropout, conv_channels=conv_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.5)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_fit, device=device)
    yt = torch.tensor(y_fit_s, device=device)
    n = len(Xt)

    loss_curve = []
    val_r2_curve = []
    best_val_r2 = -1e9
    best_state = None
    epochs_since_improve = 0
    t0 = time.perf_counter()
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
        sched.step()
        epoch_loss = total_loss / n
        loss_curve.append(epoch_loss)

        if early_stop:
            model.eval()
            val_pred = predict_batched(model, X_eval, device) * y_std + y_mean
            val_r2 = evaluate(y_eval, val_pred)["r2"]
            val_r2_curve.append(val_r2)
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            msg = f"[{tag}] epoch {epoch+1}/{epochs} train_mse(scaled)={epoch_loss:.4f} lr={sched.get_last_lr()[0]:.2e}"
            if early_stop:
                msg += f" val_r2={val_r2_curve[-1]:.4f} best={best_val_r2:.4f}"
            print(msg)

        if early_stop and epochs_since_improve >= patience:
            print(f"[{tag}] early stop at epoch {epoch+1} (no val improvement in {patience} epochs), best_val_r2={best_val_r2:.4f}")
            break
    train_seconds = time.perf_counter() - t0

    if early_stop and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    pred_eval_s = predict_batched(model, X_eval, device)
    pred_eval = pred_eval_s * y_std + y_mean  # inverse-transform to real AQI units
    check_not_degenerate(pred_eval, f"CNN-LSTM[{tag}]")
    metrics = evaluate(y_eval, pred_eval)
    metrics["train_seconds"] = train_seconds
    metrics["seq_len"] = seq_len
    metrics["epochs"] = epochs
    metrics["lr"] = lr
    metrics["hidden"] = hidden
    metrics["n_train_seq"] = int(len(X_fit))
    metrics["loss_curve"] = loss_curve
    metrics["val_r2_curve"] = val_r2_curve
    metrics["best_val_epoch"] = (len(val_r2_curve) - epochs_since_improve) if early_stop else None

    if final_test_check:
        X_test, y_test, meta_test = prep_x(test_df)
        pred_test_s = predict_batched(model, X_test, device)
        pred_test = pred_test_s * y_std + y_mean
        test_metrics = evaluate(y_test, pred_test)
        metrics["true_test"] = test_metrics
        metrics["true_test_meta"] = meta_test

    return metrics, model, scaler_x, y_mean, y_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "final", "earlystop"], default="sweep")
    ap.add_argument("--seq_len", type=int, default=14)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--head_dropout", type=float, default=0.0)
    ap.add_argument("--conv_channels", type=int, default=32)
    ap.add_argument("--out_tag", type=str, default="earlystop")
    args = ap.parse_args()

    df, cols = load_base()

    if args.mode == "sweep":
        results = []
        for seq_len in [7, 14, 28]:
            print(f"\n=== sweep seq_len={seq_len} ===")
            m, *_ = run_config(df, cols, seq_len, epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                                use_val_split=True, tag=f"seq{seq_len}")
            results.append(m)
            print(f"[seq{seq_len}] VAL r2={m['r2']:.4f} rmse={m['rmse']:.2f} n={m['n']} loss_last5={m['loss_curve'][-5:]}")
        with open(os.path.join(TUNE_DIR, "seq_len_sweep.json"), "w") as f:
            json.dump(results, f, indent=2)
        best = max(results, key=lambda m: m["r2"])
        print(f"\nBEST seq_len={best['seq_len']} val_r2={best['r2']:.4f}")
    elif args.mode == "earlystop":
        print(f"\n=== EARLYSTOP run seq_len={args.seq_len} max_epochs={args.epochs} lr={args.lr} "
              f"hidden={args.hidden} weight_decay={args.weight_decay} patience={args.patience} ===")
        m, model, scaler_x, y_mean, y_std = run_config(
            df, cols, args.seq_len, epochs=args.epochs, lr=args.lr, hidden=args.hidden,
            use_val_split=True, tag="es", weight_decay=args.weight_decay,
            early_stop=True, patience=args.patience, final_test_check=True,
            num_layers=args.num_layers, dropout=args.dropout, head_dropout=args.head_dropout,
            conv_channels=args.conv_channels)
        print(f"[EARLYSTOP] best_val_epoch={m['best_val_epoch']} best_val_r2={max(m['val_r2_curve']):.4f}")
        print(f"[EARLYSTOP] TRUE TEST r2={m['true_test']['r2']:.4f} rmse={m['true_test']['rmse']:.2f} "
              f"mae={m['true_test']['mae']:.2f} n={m['true_test']['n']}")
        m.pop("true_test_meta", None)
        tag = args.out_tag
        with open(os.path.join(TUNE_DIR, f"{tag}_metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        import joblib
        torch.save(model.state_dict(), os.path.join(TUNE_DIR, f"{tag}_model.pt"))
        joblib.dump(scaler_x, os.path.join(TUNE_DIR, f"{tag}_scaler_x.pkl"))
        with open(os.path.join(TUNE_DIR, f"{tag}_meta.json"), "w") as f:
            json.dump({"seq_len": args.seq_len, "hidden": args.hidden, "y_mean": y_mean, "y_std": y_std,
                       "cols": cols, "num_layers": args.num_layers, "dropout": args.dropout,
                       "head_dropout": args.head_dropout, "conv_channels": args.conv_channels}, f, indent=2)
    else:
        print(f"\n=== FINAL run seq_len={args.seq_len} epochs={args.epochs} lr={args.lr} hidden={args.hidden} ===")
        m, *_ = run_config(df, cols, args.seq_len, epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                            use_val_split=False, tag="final")
        print(f"[FINAL] TEST r2={m['r2']:.4f} rmse={m['rmse']:.2f} mae={m['mae']:.2f} n={m['n']}")
        with open(os.path.join(TUNE_DIR, "final_tuned_metrics.json"), "w") as f:
            json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
