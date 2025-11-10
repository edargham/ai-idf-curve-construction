import os
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import random

from shared_io import (
    build_idf_from_out_scaled,
    compute_and_save_duration_metrics,
    plot_idf_comparisons,
    plot_predictions_vs_observations,
)
from shared_metrics import nash_sutcliffe_efficiency, squared_pearson_r2 as r2_score
from shared_dataprep_tcn import create_feats_and_labels, TensorRegressionDataset

# Reproducibility: set a global seed and stabilize RNGs
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    torch.manual_seed(SEED)
    # If on CUDA, set all GPU seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # Prefer deterministic algorithms when possible; may raise on some backends
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Fallback for CUDA cuDNN
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
except Exception:
    # If torch isn't available or fails, continue but reproducibility may be limited
    pass

input_file = os.path.join(
    os.path.dirname(__file__), "..", "results", "annual_max_intensity.csv"
)

df = pd.read_csv(input_file)

(
    (X_train_scaled, y_train_scaled),
    (X_val_scaled, y_val_scaled),
    (scaler_X, scaler_y),
    (train_df_combined, val_df_combined),
    combined_df,
    duration_minutes,
    col_names,
    shared,
) = create_feats_and_labels(df)

# Replace SVR with a PyTorch TCAN, using MPS when available and 5-fold cross-validation
# Device selection: prefer MPS on macOS, fallback to CUDA or CPU (torch already imported at top)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Reproducible shuffling generator for DataLoaders used in training
train_loader_gen = torch.Generator()
train_loader_gen.manual_seed(SEED)

class MHSA1d(nn.Module):
    """Multi-Head Self-Attention over 1D feature maps.
    
    Treats the length dimension (L) as the token axis and the channel dimension (C)
    as the embedding size. Input/Output shape: (B, C, L).
    """
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.05):
        super().__init__()
        # Ensure num_heads divides channels; if not, fallback to the largest divisor <= num_heads
        nh = max(1, min(num_heads, channels))
        for h in range(min(num_heads, channels), 0, -1):
            if channels % h == 0:
                nh = h
                break
        self.num_heads = nh
        self.embed_dim = channels
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, L, C)
        x_seq = x.permute(0, 2, 1)
        x_norm = self.ln1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x_res = x_seq + self.dropout(attn_out)
        x_res = self.ln2(x_res)
        # Back to (B, C, L)
        return x_res.permute(0, 2, 1)


class TemporalBlock(nn.Module):
    """Simplified temporal block with residual convs and optional self-attention."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.01, attn_heads: int = 0, attn_dropout: float = 0.05):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) // 2  # same padding

        # Two conv layers with batch norm
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

        # Optional attention after residual fusion
        self.attn = MHSA1d(out_channels, num_heads=attn_heads, dropout=attn_dropout) if attn_heads and attn_heads > 0 else None

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu_out(out + identity)

        # Self-attention refinement (keeps shape)
        if self.attn is not None:
            out = self.attn(out)

        return out


class TCAN(nn.Module):
    def __init__(self, seq_len=3, num_filters=144, kernel_size=3, dropout=0.01, num_levels=2, attn_heads: int = 8, attn_dropout: float = 0.05):
        super(TCAN, self).__init__()
        self.seq_len = seq_len
        
        # Simplified 2-level architecture with gradual channel expansion
        layers = []
        in_channels = 1
        for i in range(num_levels):
            out_channels = num_filters * (2 ** i)
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, dropout, attn_heads=attn_heads, attn_dropout=attn_dropout
            ))
            in_channels = out_channels
        
        self.network = nn.Sequential(*layers)
        
        # Global average pooling for consistent output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Simplified fully connected layers with minimal dropout
        final_channels = num_filters * (2 ** (num_levels - 1))
        self.fc1 = nn.Linear(final_channels, final_channels * 2)
        self.bn_fc1 = nn.BatchNorm1d(final_channels * 2)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout * 0.1)
        
        self.fc2 = nn.Linear(final_channels * 2, final_channels)
        self.bn_fc2 = nn.BatchNorm1d(final_channels)
        self.relu_fc2 = nn.ReLU()
        
        self.fc3 = nn.Linear(final_channels, 1)

    def forward(self, x):
        # x shape: (batch_size, 1, seq_len) or (batch_size, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # add channel dimension
        
        # Temporal feature extraction
        x = self.network(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        
        x = self.fc3(x)
        
        return x

# Prepare arrays
X_train_arr = X_train_scaled
y_train_arr = y_train_scaled.reshape(
    -1,
)

X_val_arr = X_val_scaled if X_val_scaled is not None and len(X_val_scaled) > 0 else None
y_val_arr = (
    y_val_scaled.reshape(
        -1,
    )
    if y_val_scaled is not None and len(y_val_scaled) > 0
    else None
)


# 5-fold cross-validation to choose hidden size
hidden_candidates = [144, 160, 176, 192]
kf = KFold(n_splits=4, shuffle=False)

# Precompute durations, ranks and IDF settings used during training/validation
return_periods = [2, 5, 10, 25, 50, 100]
frequency_factors = {2: 0.85, 5: 1.00, 10: 1.25, 25: 1.50, 50: 1.75, 100: 2.00}

durations = np.linspace(1, 1440, 1440)
min_rank = combined_df["weibull_rank"].min()
max_rank = combined_df["weibull_rank"].max()
np.random.seed(SEED)
random_ranks = np.random.uniform(min_rank, max_rank, size=durations.shape)
ranks = random_ranks

standard_durations_minutes = [
    5,
    10,
    15,
    30,
    60,
    90,
    120,
    180,
    360,
    720,
    900,
    1080,
    1440,
]


def train_epoch(model, loader, opt, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def eval_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.vstack(preds).flatten() if preds else np.array([])
    trues = np.vstack(trues).flatten() if trues else np.array([])
    return total_loss / len(loader.dataset), preds, trues


print(
    "Starting 5-fold cross-validation on training split (1998-2018) to select hidden size..."
)
cv_results = {}
for h in hidden_candidates:
    fold_losses = []
    for train_idx, val_idx in kf.split(X_train_arr):
        X_tr, X_va = X_train_arr[train_idx], X_train_arr[val_idx]
        y_tr, y_va = y_train_arr[train_idx], y_train_arr[val_idx]

        ds_tr = TensorRegressionDataset(X_tr, y_tr)
        ds_va = TensorRegressionDataset(X_va, y_va)
        loader_tr = DataLoader(
            ds_tr, batch_size=64, shuffle=True, generator=train_loader_gen
        )
        loader_va = DataLoader(ds_va, batch_size=64, shuffle=False)

        # For the TCAN we treat the feature vector as a short sequence (seq_len = n_features)
        seq_len = X_tr.shape[1]
        model_cv = TCAN(seq_len=seq_len, num_filters=h, kernel_size=3, dropout=0.01, num_levels=2, attn_heads=4, attn_dropout=0.1).to(device)
        opt = torch.optim.AdamW(model_cv.parameters(), lr=6e-4, weight_decay=1e-6)
        criterion = nn.MSELoss()

        # quick train for a small number of epochs
        for epoch in range(300):
            train_epoch(model_cv, loader_tr, opt, criterion)

        val_loss, _, _ = eval_model(model_cv, loader_va, criterion)
        fold_losses.append(val_loss)

    cv_results[h] = np.mean(fold_losses)
    print(f"Hidden {h} -> CV MSE: {cv_results[h]:.6f}")

best_hidden = min(cv_results, key=cv_results.get)
print(f"Best hidden size from CV: {best_hidden}")


# Train final model on the full training split and evaluate on year-based validation after each epoch
final_model = TCAN(seq_len=X_train_arr.shape[1], num_filters=best_hidden, kernel_size=3, dropout=0.01, num_levels=2, attn_heads=4, attn_dropout=0.1).to(
    device
)
opt = torch.optim.AdamW(final_model.parameters(), lr=6e-4, weight_decay=1e-6)
criterion = nn.MSELoss()
# Use cosine annealing with warm restarts for better convergence
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, T_0=100, T_mult=2, eta_min=1e-7, verbose=True
)
ds_full = TensorRegressionDataset(X_train_arr, y_train_arr)
loader_full = DataLoader(
    ds_full, batch_size=64, shuffle=True, generator=train_loader_gen
)

# Validation DataLoader (year-split)
if X_val_arr is not None and len(X_val_arr) > 0:
    ds_val = TensorRegressionDataset(X_val_arr, y_val_arr)
    loader_val = DataLoader(ds_val, batch_size=64, shuffle=False)
else:
    loader_val = None

num_epochs = 600
history = {"epoch": [], "train_loss": [], "lr": [], "val_rmse": [], "val_mae": [], "val_r2": [], "val_nse": []}

# Early stopping parameters
patience = 200
patience_counter = 0

# In-memory best-weights caching (do not checkpoint to disk)
# We'll track the best validation RMSE and cache the model's state_dict in RAM
best_val_rmse = np.inf
best_weights = None
best_epoch = -1
best_metrics = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "nse": np.nan}

for epoch in range(1, num_epochs + 1):
    tr_loss = train_epoch(final_model, loader_full, opt, criterion)
    
    # Step the cosine annealing scheduler after each epoch
    if scheduler is not None:
        scheduler.step()

    # Evaluate on year-based validation split every epoch
    if loader_val is not None:
        val_loss, val_preds_scaled, val_trues_scaled = eval_model(
            final_model, loader_val, criterion
        )

        # Inverse transform scaled log-intensities back to intensities
        if len(val_preds_scaled) > 0:
            val_preds_log = scaler_y.inverse_transform(
                val_preds_scaled.reshape(-1, 1)
            ).flatten()
            y_pred_intensity = np.exp(val_preds_log)
            val_trues_log = scaler_y.inverse_transform(
                val_trues_scaled.reshape(-1, 1)
            ).flatten()
            y_val_intensity = np.exp(val_trues_log)
        else:
            y_pred_intensity = np.array([])
            y_val_intensity = np.array([])

        # Compute global metrics
        if len(y_val_intensity) > 0 and len(y_pred_intensity) == len(y_val_intensity):
            rmse = np.sqrt(mean_squared_error(y_val_intensity, y_pred_intensity))
            mae = mean_absolute_error(y_val_intensity, y_pred_intensity)
            try:
                r2_model = r2_score(y_val_intensity, y_pred_intensity)
            except Exception:
                r2_model = np.nan
            nse = nash_sutcliffe_efficiency(y_val_intensity, y_pred_intensity)
        else:
            rmse = np.nan
            mae = np.nan
            r2_model = np.nan
            nse = np.nan

        # -------------------------
        # Build IDF curves from current model predictions and compute per-duration metrics
        # Use same procedure as final IDF generation but using the model's current weights
        # -------------------------
        # Prepare features for many durations and random ranks
        log_durations = np.log(durations).reshape(-1, 1)
        # reuse the same random seed/ranks generation for reproducibility
        log_ranks = np.log(ranks).reshape(-1, 1)
        features = np.column_stack((log_durations, log_ranks))
        features_scaled_epoch = scaler_X.transform(np.asarray(features))

        # Predict with current model
        final_model.eval()
        with torch.no_grad():
            feats_t = torch.from_numpy(features_scaled_epoch.astype(np.float32)).to(
                device
            )
            out_scaled_epoch = final_model(feats_t).cpu().numpy().flatten()

        base_log_intensities_epoch = scaler_y.inverse_transform(
            out_scaled_epoch.reshape(-1, 1)
        ).flatten()
        base_intensities_epoch = np.exp(base_log_intensities_epoch)

        idf_curves_epoch = {}
        for return_period in return_periods:
            idf_curves_epoch[return_period] = (
                base_intensities_epoch * frequency_factors[return_period]
            )

        standard_idf_curves_epoch = {}
        for return_period in return_periods:
            standard_intensities = []
            for duration in standard_durations_minutes:
                duration_idx = np.abs(durations - duration).argmin()
                standard_intensities.append(
                    idf_curves_epoch[return_period][duration_idx]
                )
            standard_idf_curves_epoch[return_period] = standard_intensities

        # Compute per-duration validation metrics using the predicted intensities on the validation split
        tcan_duration_metrics_epoch = {}
        if not val_df_combined.empty and len(y_pred_intensity) > 0:
            # attach predictions to val_df_combined copy
            val_df_combined_epoch = val_df_combined.copy()
            val_df_combined_epoch["pred_intensity_epoch"] = y_pred_intensity
            for dmin, col in zip(duration_minutes, col_names):
                mask = val_df_combined_epoch["duration"] == dmin
                obs = val_df_combined_epoch.loc[mask, "intensity"].values
                preds = val_df_combined_epoch.loc[mask, "pred_intensity_epoch"].values
                if len(obs) == 0:
                    tcan_duration_metrics_epoch[col] = {
                        "R2": np.nan,
                        "NSE": np.nan,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                    }
                    continue
                try:
                    r2m = r2_score(obs, preds)
                except Exception:
                    r2m = np.nan
                nse_m = nash_sutcliffe_efficiency(obs, preds)
                mae_m = mean_absolute_error(obs, preds)
                rmse_m = np.sqrt(mean_squared_error(obs, preds))
                tcan_duration_metrics_epoch[col] = {
                    "R2": r2m,
                    "NSE": nse_m,
                    "MAE": mae_m,
                    "RMSE": rmse_m,
                }
        else:
            for col in col_names:
                tcan_duration_metrics_epoch[col] = {
                    "R2": np.nan,
                    "NSE": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                }

        # (Optional) store or log epoch per-duration metrics if desired - here we don't persist them each epoch to disk
        # -------------------------

        current_lr = opt.param_groups[0]["lr"] if opt is not None and len(opt.param_groups) > 0 else np.nan
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["lr"].append(current_lr)
        history["val_rmse"].append(rmse)
        history["val_mae"].append(mae)
        history["val_r2"].append(r2_model)
        history["val_nse"].append(nse)

        # Cache model weights in memory if this epoch improved validation RMSE
        try:
            if np.isfinite(rmse) and rmse < best_val_rmse:
                # Save a CPU copy of the state dict (safe across devices)
                best_weights = {
                    k: v.cpu().clone() for k, v in final_model.state_dict().items()
                }
                best_val_rmse = rmse
                best_epoch = epoch
                best_metrics = {"rmse": rmse, "mae": mae, "r2": r2_model, "nse": nse}
                patience_counter = 0  # Reset patience counter
                print(f"New best model cached at epoch {epoch} (val_rmse={rmse:.4f}, val_nse={nse:.4f})")
            else:
                patience_counter += 1
        except Exception:
            # If for some reason caching fails, continue training without interrupting
            print(
                "Warning: failed to cache best model weights in memory for epoch", epoch
            )
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
            break

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{num_epochs} - lr: {current_lr:.6f} - train_loss: {tr_loss:.6f} - val_rmse: {rmse:.4f} - val_mae: {mae:.4f} - val_r2: {r2_model:.4f} - val_nse: {nse:.4f}"
            )

    else:
        if epoch % 50 == 0 or epoch == 1:
            current_lr = opt.param_groups[0]["lr"] if opt is not None and len(opt.param_groups) > 0 else np.nan
            history["epoch"].append(epoch)
            history["train_loss"].append(tr_loss)
            history["lr"].append(current_lr)
            print(
                f"Epoch {epoch}/{num_epochs} - lr: {current_lr:.6f} - train_loss: {tr_loss:.6f} (no year-split validation available)"
            )


# After training, generate IDF curves using the trained PyTorch model (same approach as before)
# If we cached a best model during training, restore those weights into the model now
if best_weights is not None:
    try:
        # Move cached weights to the active device before loading
        best_state = {k: v.to(device) for k, v in best_weights.items()}
        final_model.load_state_dict(best_state)
        print(
            f"Restored best model weights from epoch {best_epoch} (val_rmse={best_val_rmse:.4f})"
        )
        # Ensure the metrics used later reflect the best epoch
        rmse = best_metrics.get("rmse", rmse)
        mae = best_metrics.get("mae", mae)
        r2_model = best_metrics.get("r2", r2_model)
        nse = best_metrics.get("nse", nse)
    except Exception:
        print(
            "Warning: failed to restore best model weights from memory; using final model weights"
        )

return_periods = [2, 5, 10, 25, 50, 100]
frequency_factors = {2: 0.85, 5: 1.00, 10: 1.25, 25: 1.50, 50: 1.75, 100: 2.00}

# Load gumbel and literature data (same as before)
gumbel_idf = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "results", "idf_data.csv")
)
literature_idf = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "results", "idf_lit.csv")
)

durations = np.linspace(1, 1440, 1440)
ranks = random_ranks
min_rank = combined_df["weibull_rank"].min()
max_rank = combined_df["weibull_rank"].max()
np.random.seed(SEED)
random_ranks = np.random.uniform(min_rank, max_rank, size=durations.shape)
ranks = random_ranks
ranks = random_ranks

log_durations = np.log(durations).reshape(-1, 1)
log_ranks = np.log(ranks).reshape(-1, 1)
features = np.column_stack((log_durations, log_ranks))
features_scaled = scaler_X.transform(np.asarray(features))

# Predict with PyTorch model (remember model outputs scaled log-intensity)

final_model.eval()
with torch.no_grad():
    feats_t = torch.from_numpy(features_scaled.astype(np.float32)).to(device)
    out_scaled = final_model(feats_t).cpu().numpy().flatten()

# Build IDF CSV and standard curves using shared verbatim logic
idf_artifacts = build_idf_from_out_scaled(
    out_scaled,
    scaler_y,
    shared["durations"],
    shared["frequency_factors"],
    shared["return_periods"],
    shared["standard_durations_minutes"],
    "idf_curves_TCAN.csv",
)

# Expose variables the rest of the file expects (keeps downstream logic unchanged)
standard_idf_curves = idf_artifacts["standard_idf_curves"]
idf_curves = idf_artifacts["idf_curves"]
base_intensities = idf_artifacts["base_intensities"]
durations = shared["durations"]
return_periods = shared["return_periods"]
frequency_factors = shared["frequency_factors"]
standard_durations_minutes = shared["standard_durations_minutes"]
duration_minutes = shared["duration_minutes"]
col_names = shared["col_names"]

tcan_duration_metrics = {}
if not val_df_combined.empty:
    # Prepare features from val_df_combined
    X_val_full = val_df_combined[["log_duration", "log_weibull_rank"]].values
    X_val_scaled_full = scaler_X.transform(X_val_full)

    # Predict with final_model
    final_model.eval()
    with torch.no_grad():
        tX = torch.from_numpy(X_val_scaled_full.astype(np.float32)).to(device)
        out_scaled_val = final_model(tX).cpu().numpy().flatten()

    preds_log = scaler_y.inverse_transform(out_scaled_val.reshape(-1, 1)).flatten()
    preds_intensity = np.exp(preds_log)

    # Use shared helper to compute and save per-duration metrics and overall metrics
    tcan_duration_metrics, overall_metrics = compute_and_save_duration_metrics(
        val_df_combined,
        preds_intensity,
        duration_minutes,
        col_names,
        "TCAN",
    )
    
    # Store overall metrics for later use
    overall_rmse_val, overall_mae_val, overall_r2_val, overall_nse_val = overall_metrics
else:
    for col in col_names:
        tcan_duration_metrics[col] = {
            "R2": np.nan,
            "NSE": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
        }

# Print concise TCAN validation metrics summary (keeps original print format)
tcan_avg_metrics = pd.DataFrame(tcan_duration_metrics).T.mean()
print("\nTCAN Validation Metrics (2019-2025) by Duration:")
print(pd.DataFrame(tcan_duration_metrics).T.round(4))
print("\nTCAN Average Metrics Across All Durations:")
print(tcan_avg_metrics.round(4))

# Define duration mapping for column names
duration_mapping = {
    0: "5 mins",
    1: "10 mins",
    2: "15 mins",
    3: "30 mins",
    4: "60 mins",
    5: "90 mins",
    6: "120 mins",
    7: "180 mins",
    8: "360 mins",
    9: "720 mins",
    10: "900 mins",
    11: "1080 mins",
    12: "1440 mins",
}

# Calculate metrics for each return period
rmse_values = []
mae_values = []
r2_values = []
nse_values = []


for rp in return_periods:
    gumbel_row = gumbel_idf[gumbel_idf["Return Period (years)"] == rp].iloc[0]

    # Extract values from gumbel data for this return period
    y_true = []
    y_pred = []

    for i, duration in enumerate(standard_durations_minutes):
        gumbel_col = duration_mapping[i]
        y_true.append(gumbel_row[gumbel_col])

        # Use the precomputed values from standard_idf_curves
        y_pred.append(standard_idf_curves[rp][i])

    # Calculate metrics
    rmse_rp = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_rp = mean_absolute_error(y_true, y_pred)
    r2_rp = r2_score(y_true, y_pred)
    nse_rp = nash_sutcliffe_efficiency(y_true, y_pred)

    rmse_values.append(rmse_rp)
    mae_values.append(mae_rp)
    r2_values.append(r2_rp)
    nse_values.append(nse_rp)

# Display metrics
metrics_df = pd.DataFrame(
    {
        "Return Period": return_periods,
        "RMSE": [round(x, 4) for x in rmse_values],
        "MAE": [round(x, 4) for x in mae_values],
        "R2": [round(x, 4) for x in r2_values],
        "NSE": [round(x, 4) for x in nse_values],
    }
)
print("\nModel Performance Metrics by Return Period:")
print(metrics_df)

# Calculate overall metrics
overall_rmse = np.mean(rmse_values)
overall_mae = np.mean(mae_values)
overall_r2 = np.mean(r2_values)
overall_nse = np.mean(nse_values)

print(f"\nOverall RMSE: {overall_rmse:.4f}")
print(f"Overall MAE: {overall_mae:.4f}")
print(f"Overall R2: {overall_r2:.4f}")
print(f"Overall NSE: {overall_nse:.4f}")

# Calculate metrics for literature comparison
# Create mapping for literature data columns
literature_duration_mapping = {
    5: "5 mins",
    10: "10 mins",
    15: "15 mins",
    30: "30 mins",
    60: "60 mins",
    90: "90 mins",
    120: "120 mins",
    180: "180 mins",
    360: "360 mins",
    720: "720 mins",
    900: "900 mins",
    1080: "1080 mins",
    1440: "1440 mins",
}

# Calculate metrics for each return period against literature data
lit_rmse_values = []
lit_mae_values = []
lit_r2_values = []
lit_nse_values = []

for rp in return_periods:
    lit_row = literature_idf[literature_idf["Return Period (years)"] == rp].iloc[0]

    # Extract values from literature data for this return period
    y_true_lit = []
    y_pred_lit = []

    for i, duration in enumerate(standard_durations_minutes):
        lit_col = literature_duration_mapping[duration]
        lit_value = lit_row[lit_col]

        # Only include non-null values from literature data
        if pd.notna(lit_value) and lit_value != "":
            y_true_lit.append(float(lit_value))
            y_pred_lit.append(standard_idf_curves[rp][i])

    # Calculate metrics only if we have data points
    if len(y_true_lit) > 0:
        lit_rmse = np.sqrt(mean_squared_error(y_true_lit, y_pred_lit))
        lit_mae = mean_absolute_error(y_true_lit, y_pred_lit)
        lit_r2 = r2_score(y_true_lit, y_pred_lit)
        lit_nse = nash_sutcliffe_efficiency(y_true_lit, y_pred_lit)
    else:
        lit_rmse = np.nan
        lit_mae = np.nan
        lit_r2 = np.nan
        lit_nse = np.nan

    lit_rmse_values.append(lit_rmse)
    lit_mae_values.append(lit_mae)
    lit_r2_values.append(lit_r2)
    lit_nse_values.append(lit_nse)

# Display literature comparison metrics (filter out NaN values)
valid_lit_metrics = [
    (rp, rmse, mae, r2, nse)
    for rp, rmse, mae, r2, nse in zip(
        return_periods, lit_rmse_values, lit_mae_values, lit_r2_values, lit_nse_values
    )
    if not np.isnan(rmse)
]

if valid_lit_metrics:
    lit_metrics_df = pd.DataFrame(
        {
            "Return Period": [x[0] for x in valid_lit_metrics],
            "RMSE": [round(x[1], 4) for x in valid_lit_metrics],
            "MAE": [round(x[2], 4) for x in valid_lit_metrics],
            "R2": [round(x[3], 4) for x in valid_lit_metrics],
            "NSE": [round(x[4], 4) for x in valid_lit_metrics],
        }
    )
    print("\nModel Performance Metrics vs Literature by Return Period:")
    print(lit_metrics_df)

    # Calculate overall metrics for literature comparison
    valid_lit_rmse = [x for x in lit_rmse_values if not np.isnan(x)]
    valid_lit_mae = [x for x in lit_mae_values if not np.isnan(x)]
    valid_lit_r2 = [x for x in lit_r2_values if not np.isnan(x)]
    valid_lit_nse = [x for x in lit_nse_values if not np.isnan(x)]

    overall_lit_rmse = np.mean(valid_lit_rmse) if valid_lit_rmse else np.nan
    overall_lit_mae = np.mean(valid_lit_mae) if valid_lit_mae else np.nan
    overall_lit_r2 = np.mean(valid_lit_r2) if valid_lit_r2 else np.nan
    overall_lit_nse = np.mean(valid_lit_nse) if valid_lit_nse else np.nan

    print("\nOverall Literature Comparison:")
    print(f"RMSE: {overall_lit_rmse:.4f}")
    print(f"MAE: {overall_lit_mae:.4f}")
    print(f"R2: {overall_lit_r2:.4f}")
    print(f"NSE: {overall_lit_nse:.4f}")

    # Save literature comparison metrics to literature_performance_metrics.csv
    lit_comparison_metrics = {
        "Model": "TCAN",
        "RMSE": overall_lit_rmse,
        "MAE": overall_lit_mae,
        "R2": overall_lit_r2,
        "NSE": overall_lit_nse,
    }

    # Check if literature performance metrics file exists
    lit_metrics_file = os.path.join(
        os.path.dirname(__file__), "..", "results", "literature_performance_metrics.csv"
    )
    if os.path.exists(lit_metrics_file):
        # Load existing metrics file and append/update
        lit_metrics_df = pd.read_csv(lit_metrics_file)

        # Check if TCAN already exists in the dataframe
        if "TCAN" in lit_metrics_df["Model"].values:
            # Update existing row
            for col, val in lit_comparison_metrics.items():
                lit_metrics_df.loc[lit_metrics_df["Model"] == "TCAN", col] = val
        else:
            # Append new row
            lit_metrics_df = pd.concat(
                [lit_metrics_df, pd.DataFrame([lit_comparison_metrics])],
                ignore_index=True,
            )
    else:
        # Create new metrics dataframe
        lit_metrics_df = pd.DataFrame([lit_comparison_metrics])

    # Save metrics to CSV
    lit_metrics_df.to_csv(lit_metrics_file, index=False)
    print(f"Literature comparison metrics saved to: {lit_metrics_file}")

# Check if performance metrics file exists, append if it does, create if not
metrics_file_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "model_performance_metrics.csv"
)
# Use model evaluation metrics (from test set) rather than the overall comparison metrics
# rmse, mae, r2_model and nse are calculated earlier from the model evaluation on test data
metrics_row = {"Model": "TCAN", "RMSE": rmse, "MAE": mae, "R2": r2_model, "NSE": nse}

if os.path.exists(metrics_file_path):
    # Load existing metrics file and append new row
    metrics_df = pd.read_csv(metrics_file_path)

    # Check if TCAN model already exists in the dataframe
    if "TCAN" in metrics_df["Model"].values:
        # Update existing row
        for col, val in metrics_row.items():
            metrics_df.loc[metrics_df["Model"] == "TCAN", col] = val
    else:
        # Append new row
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([metrics_row])], ignore_index=True
        )
else:
    # Create new metrics dataframe
    metrics_df = pd.DataFrame([metrics_row])

# Save metrics to CSV
metrics_df.to_csv(metrics_file_path, index=False)
print(f"Model performance metrics saved to: {metrics_file_path}")


# Central plotting call (verbatim logic moved to shared_io.plot_idf_comparisons)
# Prefer raw validation metrics (rmse, mae, r2_model, nse) computed during training
# for annotating the plots. Fall back to comparison aggregates when not available.
try:
    plot_overall_metrics = (rmse, mae, r2_model, nse)
except NameError:
    plot_overall_metrics = (overall_rmse, overall_mae, overall_r2, overall_nse)

try:
    overall_lit_metrics = (
        overall_lit_rmse,
        overall_lit_mae,
        overall_lit_r2,
        overall_lit_nse,
    )
except NameError:
    overall_lit_metrics = None

plot_idf_comparisons(
    standard_idf_curves,
    standard_durations_minutes,
    return_periods,
    gumbel_idf,
    literature_idf,
    duration_mapping,
    literature_duration_mapping,
    # model_metrics (raw validation) vs gumbel_metrics (comparison aggregates)
    plot_overall_metrics,
    (overall_rmse, overall_mae, overall_r2, overall_nse),
    overall_lit_metrics,
    "TCAN",
    "tcan",
)

# Plot predictions vs observations after IDF comparisons
if not val_df_combined.empty and 'overall_rmse_val' in locals():
    y_val_intensity = val_df_combined['intensity'].values
    X_val_full = val_df_combined[["log_duration", "log_weibull_rank"]].values
    X_val_scaled_full = scaler_X.transform(X_val_full)
    final_model.eval()
    with torch.no_grad():
        out_scaled_val = final_model(torch.from_numpy(X_val_scaled_full.astype(np.float32)).to(device)).cpu().numpy().flatten()
    preds_log = scaler_y.inverse_transform(out_scaled_val.reshape(-1, 1)).flatten()
    preds_intensity = np.exp(preds_log)

    overall_rmse, overall_mae, overall_r2, overall_nse = plot_overall_metrics
    
    metrics = {
        'rmse': overall_rmse,
        'mae': overall_mae,
        'r2': overall_r2,
        'nse': overall_nse
    }

    plot_predictions_vs_observations(y_val_intensity, preds_intensity, 'TCAN', 'tcan', metrics)
