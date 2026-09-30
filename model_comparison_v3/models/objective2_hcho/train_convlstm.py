"""ConvLSTM spatio-temporal model for Objective 2 - predicts next month's
national HCHO map from the last N=6 monthly multi-channel maps (HCHO, fire
count, wind u/v). Full national grid (all 4,447 real India cells), same
scope as the statistical baseline, for a fair comparison. GPU-trained.

PyTorch has no built-in ConvLSTM layer - this is a standard ConvLSTMCell
(Shi et al. 2015 formulation: a single Conv2d producing the 4 LSTM gates
from the concatenated input+hidden state), stacked into a small sequence
model with a final Conv2d head collapsing hidden channels to 1 output
channel (predicted HCHO).

Masked MSE loss: only real India cells contribute (via the same mask used
in the statistical baseline), and target months with zero valid HCHO
coverage (the 2 flagged empty months) are excluded from training entirely,
not silently treated as zero.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts", "convlstm")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

sys.path.insert(0, THIS_DIR)
from eval_utils import evaluate_masked, timer, check_not_degenerate  # noqa: E402

LOOKBACK = 6
HIDDEN_CHANNELS = 16
N_CHANNELS = 4  # hcho, fire_count, wind_u, wind_v
EPOCHS = 60
LR = 1e-3


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, H: int, W: int):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels)
        self.hidden_channels = hidden_channels
        self.H, self.W = H, W
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_len, channels, H, W)
        b, seq_len = x.shape[0], x.shape[1]
        h = torch.zeros(b, self.hidden_channels, self.H, self.W, device=x.device)
        c = torch.zeros(b, self.hidden_channels, self.H, self.W, device=x.device)
        for t in range(seq_len):
            h, c = self.cell(x[:, t], h, c)
        return self.head(h).squeeze(1)  # (batch, H, W)


def build_sequences(tensor: np.ndarray, mask: np.ndarray, target_indices: list[int], lookback: int = LOOKBACK):
    """tensor: (n_months, C, H, W), already NaN-filled-with-0 for model input
    (mask handles which cells matter for loss/eval - the model still sees 0
    at masked-out ocean/non-India cells, which is fine, they contribute
    nothing to the loss regardless)."""
    X, y, valid_targets = [], [], []
    for i in target_indices:
        if i - lookback < 0:
            continue
        window = tensor[i - lookback:i]  # (lookback, C, H, W)
        target = tensor[i, 0]  # HCHO channel
        if np.isnan(target[mask]).all():
            continue  # empty month - excluded, not zero-filled
        X.append(window)
        y.append(target)
        valid_targets.append(i)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), valid_targets


def main():
    data = np.load(os.path.join(DATA_DIR, "spatiotemporal_grid.npz"))
    tensor_raw, mask = data["tensor"], data["mask"]
    with open(os.path.join(DATA_DIR, "spatiotemporal_grid_manifest.json")) as f:
        manifest = json.load(f)
    split_idx = manifest["split_idx"]
    n_months, C, H, W = tensor_raw.shape

    # per-channel z-score normalization (fit on train months only), then 0-fill for model input
    tensor_norm = tensor_raw.copy()
    means, stds = [], []
    for ch in range(C):
        train_vals = tensor_raw[:split_idx, ch][np.broadcast_to(mask, (split_idx, H, W))]
        m, s = np.nanmean(train_vals), np.nanstd(train_vals) + 1e-12
        means.append(m); stds.append(s)
        tensor_norm[:, ch] = (tensor_raw[:, ch] - m) / s
    tensor_norm = np.nan_to_num(tensor_norm, nan=0.0)

    train_targets = list(range(LOOKBACK, split_idx))
    test_targets = list(range(split_idx, n_months))
    X_train, y_train_norm, train_idx = build_sequences(tensor_norm, mask, train_targets)
    X_test, y_test_norm, test_idx = build_sequences(tensor_norm, mask, test_targets)
    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")
    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError("Not enough monthly sequences to build ConvLSTM windows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    model = ConvLSTMPredictor(N_CHANNELS, HIDDEN_CHANNELS, H, W).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    mask_t = torch.tensor(mask, device=device, dtype=torch.bool)
    Xt = torch.tensor(X_train, device=device)
    yt = torch.tensor(y_train_norm, device=device)
    yt = torch.nan_to_num(yt, nan=0.0)  # target normalized HCHO, masked loss ignores non-India/NaN cells anyway via mask_t

    def masked_mse(pred, target):
        diff = (pred - target) ** 2
        return diff[:, mask_t].mean()

    with timer() as t_train:
        for epoch in range(EPOCHS):
            model.train()
            opt.zero_grad()
            pred = model(Xt)
            loss = masked_mse(pred, yt)
            loss.backward()
            opt.step()
            if (epoch + 1) % 10 == 0:
                print(f"epoch {epoch+1}/{EPOCHS} train_masked_mse={loss.item():.6f}")

    model.eval()
    with timer() as t_infer:
        with torch.no_grad():
            pred_test_norm = model(torch.tensor(X_test, device=device)).cpu().numpy()
    with torch.no_grad():
        pred_train_norm = model(Xt).cpu().numpy()

    hcho_mean, hcho_std = means[0], stds[0]
    pred_test = pred_test_norm * hcho_std + hcho_mean
    pred_train = pred_train_norm * hcho_std + hcho_mean
    y_test_true = tensor_raw[test_idx, 0]
    y_train_true = tensor_raw[train_idx, 0]

    check_not_degenerate(pred_test, np.broadcast_to(mask, pred_test.shape), "ConvLSTM")

    metrics = {
        "train": evaluate_masked(y_train_true, pred_train, mask),
        "test": evaluate_masked(y_test_true, pred_test, mask),
        "train_seconds": t_train["seconds"],
        "inference_seconds_total": t_infer["seconds"],
        "inference_seconds_per_month": t_infer["seconds"] / len(X_test),
        "n_train_sequences": len(X_train),
        "n_test_sequences": len(X_test),
        "lookback_months": LOOKBACK,
        "hidden_channels": HIDDEN_CHANNELS,
        "device": device,
    }
    print(json.dumps(metrics, indent=2))

    torch.save(model.state_dict(), os.path.join(ARTIFACTS_DIR, "convlstm.pt"))
    np.savez_compressed(os.path.join(ARTIFACTS_DIR, "predictions.npz"), preds=pred_test, targets=y_test_true, test_month_idx=test_idx)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "norm_stats.json"), "w") as f:
        json.dump({"means": [float(m) for m in means], "stds": [float(s) for s in stds]}, f, indent=2)

    # channel-ablation "feature importance" substitute (Phase 2): retrain-free
    # approximation - zero out one input channel at inference and measure R2 drop
    base_r2 = evaluate_masked(y_test_true, pred_test, mask)["r2"]
    ablation = []
    for ch, name in enumerate(["hcho", "fire_count", "wind_u", "wind_v"]):
        X_ablated = X_test.copy()
        X_ablated[:, :, ch] = 0.0
        with torch.no_grad():
            pred_ablated_norm = model(torch.tensor(X_ablated, device=device)).cpu().numpy()
        pred_ablated = pred_ablated_norm * hcho_std + hcho_mean
        r2_drop = base_r2 - evaluate_masked(y_test_true, pred_ablated, mask)["r2"]
        ablation.append({"channel": name, "r2_drop_when_zeroed": r2_drop})
    import pandas as pd
    pd.DataFrame(ablation).sort_values("r2_drop_when_zeroed", ascending=False).to_csv(
        os.path.join(ARTIFACTS_DIR, "channel_ablation.csv"), index=False)

    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
