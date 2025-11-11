import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Additional imports required by the ANN script
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from split_utils import build_train_val
from shared_io import shared_preprocessing, build_idf_from_out_scaled, compute_and_save_duration_metrics, plot_idf_comparisons, plot_predictions_vs_observations
from shared_metrics import nash_sutcliffe_efficiency, squared_pearson_r2 as r2_score
from uncertainty_analysis import analyze_ai_model_uncertainty

# Reproducibility: set a global seed and stabilize RNGs
SEED = 368683
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
except Exception:
    pass

# Use centralized preprocessing (verbatim copy lives in shared_io.shared_preprocessing)
shared = shared_preprocessing()
# expose variables expected by the rest of the script without changing logic
df = shared['df']
combined_df = shared['combined_df']
duration_minutes = shared['duration_minutes']
col_names = shared['col_names']

# Use the centralized year-split so train/val are exactly as before
train_df_combined, val_df_combined, years = build_train_val(df)

# Transform (log) for modeling (kept identical to originals)
train_df_combined['log_duration'] = np.log(train_df_combined['duration'])
train_df_combined['log_weibull_rank'] = np.log(train_df_combined['weibull_rank'])
train_df_combined['log_intensity'] = np.log(train_df_combined['intensity'])

if not val_df_combined.empty:
    val_df_combined['log_duration'] = np.log(val_df_combined['duration'])
    val_df_combined['log_weibull_rank'] = np.log(val_df_combined['weibull_rank'])
    val_df_combined['log_intensity'] = np.log(val_df_combined['intensity'])

# Prepare X/y (identical logic to other model scripts)
X_train = train_df_combined[['log_duration','log_weibull_rank']]
y_train = train_df_combined['log_intensity']

X_val = val_df_combined[['log_duration','log_weibull_rank']] if not val_df_combined.empty else None
y_val = val_df_combined['log_intensity'] if not val_df_combined.empty else None

# Standard scale the data (fit on train only)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_arr = np.asarray(X_train)
X_train_scaled = scaler_X.fit_transform(X_train_arr)
X_val_scaled = scaler_X.transform(np.asarray(X_val)) if X_val is not None and len(X_val)>0 else None

y_train_scaled = scaler_y.fit_transform(np.asarray(y_train).reshape(-1,1)).flatten()
y_val_scaled = scaler_y.transform(np.asarray(y_val).reshape(-1,1)).flatten() if y_val is not None and len(y_val)>0 else None

# Device selection and reproducible DataLoader generator
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_loader_gen = torch.Generator()
train_loader_gen.manual_seed(SEED)

class TensorRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max(1, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(1, hidden_size // 2), 1)
        )

    def forward(self, x):
        return self.net(x)


# Prepare arrays
X_train_arr = X_train_scaled
y_train_arr = y_train_scaled.reshape(-1,)

X_val_arr = X_val_scaled if X_val_scaled is not None and len(X_val_scaled)>0 else None
y_val_arr = y_val_scaled.reshape(-1,) if y_val_scaled is not None and len(y_val_scaled)>0 else None


# 5-fold cross-validation to choose hidden size
hidden_candidates = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

kf = KFold(n_splits=10, shuffle=False)

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


print('Starting 5-fold cross-validation on training split (1998-2018) to select hidden size...')
cv_results = {}
for h in hidden_candidates:
    fold_losses = []
    for train_idx, val_idx in kf.split(X_train_arr):
        X_tr, X_va = X_train_arr[train_idx], X_train_arr[val_idx]
        y_tr, y_va = y_train_arr[train_idx], y_train_arr[val_idx]

        ds_tr = TensorRegressionDataset(X_tr, y_tr)
        ds_va = TensorRegressionDataset(X_va, y_va)
        # Training loader: deterministic shuffle with seeded generator
        loader_tr = DataLoader(ds_tr, batch_size=64, shuffle=True, generator=train_loader_gen)
        # Validation loader: do not shuffle
        loader_va = DataLoader(ds_va, batch_size=64, shuffle=False)

        model_cv = MLP(input_dim=X_tr.shape[1], hidden_size=h).to(device)
        opt = torch.optim.AdamW(model_cv.parameters(), lr=1.25e-3)
        criterion = nn.MSELoss()

        # quick train for a small number of epochs
        for epoch in range(60):
            train_epoch(model_cv, loader_tr, opt, criterion)

        val_loss, _, _ = eval_model(model_cv, loader_va, criterion)
        fold_losses.append(val_loss)

    cv_results[h] = np.mean(fold_losses)
    print(f'Hidden {h} -> CV MSE: {cv_results[h]:.6f}')

best_hidden = min(cv_results, key=cv_results.get)
print(f'Best hidden size from CV: {best_hidden}')


# Train final model on the full training split and evaluate on year-based validation after each epoch
final_model = MLP(input_dim=X_train_arr.shape[1], hidden_size=best_hidden).to(device)
opt = torch.optim.AdamW(final_model.parameters(), lr=1.25e-3)
criterion = nn.MSELoss()
# Configure ReduceLROnPlateau with a minimum lr floor and verbose logging
min_lr = 1e-6
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100, verbose=True, min_lr=min_lr)
ds_full = TensorRegressionDataset(X_train_arr, y_train_arr)
# Final training loader: deterministic shuffle with seeded generator
loader_full = DataLoader(ds_full, batch_size=64, shuffle=True, generator=train_loader_gen)

# Validation DataLoader (year-split)
if X_val_arr is not None and len(X_val_arr) > 0:
    ds_val = TensorRegressionDataset(X_val_arr, y_val_arr)
    # Validation loader: do not shuffle
    loader_val = DataLoader(ds_val, batch_size=64, shuffle=False)
else:
    loader_val = None

num_epochs = 500
history = {'epoch': [], 'val_rmse': [], 'val_mae': [], 'val_r2': [], 'val_nse': []}

# In-memory best-weights caching (do not checkpoint to disk)
# We'll track the best validation RMSE and cache the model's state_dict in RAM
best_val_rmse = np.inf
best_weights = None
best_epoch = -1
best_metrics = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'nse': np.nan}

for epoch in range(1, num_epochs+1):
    tr_loss = train_epoch(final_model, loader_full, opt, criterion)

    # Evaluate on year-based validation split every epoch
    if loader_val is not None:
        val_loss, val_preds_scaled, val_trues_scaled = eval_model(final_model, loader_val, criterion)

        if scheduler is not None:
            # Step the scheduler with the validation loss (ReduceLROnPlateau expects a scalar)
            try:
                prev_lrs = [group['lr'] for group in opt.param_groups]
                scheduler.step(val_loss)
                new_lrs = [group['lr'] for group in opt.param_groups]
                if prev_lrs != new_lrs:
                    print(f"Learning rate reduced: {prev_lrs} -> {new_lrs}")
            except Exception:
                print('Warning: scheduler.step failed')

        # Inverse transform scaled log-intensities back to intensities
        if len(val_preds_scaled) > 0:
            val_preds_log = scaler_y.inverse_transform(val_preds_scaled.reshape(-1,1)).flatten()
            y_pred_intensity = np.exp(val_preds_log)
            val_trues_log = scaler_y.inverse_transform(val_trues_scaled.reshape(-1,1)).flatten()
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

        history['epoch'].append(epoch)
        history['val_rmse'].append(rmse)
        history['val_mae'].append(mae)
        history['val_r2'].append(r2_model)
        history['val_nse'].append(nse)

        # Cache model weights in memory if this epoch improved validation RMSE
        try:
            if np.isfinite(rmse) and rmse < best_val_rmse:
                # Save a CPU copy of the state dict (safe across devices)
                best_weights = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
                best_val_rmse = rmse
                best_epoch = epoch
                best_metrics = {'rmse': rmse, 'mae': mae, 'r2': r2_model, 'nse': nse}
                print(f"New best model cached at epoch {epoch} (val_rmse={rmse:.4f})")
        except Exception:
            # If for some reason caching fails, continue training without interrupting
            print('Warning: failed to cache best model weights in memory for epoch', epoch)

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{num_epochs} - train_loss: {tr_loss:.6f} - val_rmse: {rmse:.4f} - val_mae: {mae:.4f} - val_r2: {r2_model:.4f} - val_nse: {nse:.4f}')

    else:
        if epoch % 50 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{num_epochs} - train_loss: {tr_loss:.6f} (no year-split validation available)')


# After training, restore best weights (if cached) and build IDF/metrics/plots via shared helpers
if best_weights is not None:
    try:
        best_state = {k: v.to(device) for k, v in best_weights.items()}
        final_model.load_state_dict(best_state)
        print(f"Restored best model weights from epoch {best_epoch} (val_rmse={best_val_rmse:.4f})")
        rmse = best_metrics.get('rmse', rmse)
        mae = best_metrics.get('mae', mae)
        r2_model = best_metrics.get('r2', r2_model)
        nse = best_metrics.get('nse', nse)
    except Exception:
        print('Warning: failed to restore best model weights from memory; using final model weights')

# Shared constants and settings (re-use module-level `shared` populated earlier)
durations = shared['durations']
frequency_factors = shared['frequency_factors']
return_periods = shared['return_periods']
standard_durations_minutes = shared['standard_durations_minutes']

# Build features for many durations and predict with ANN
log_durations = np.log(durations).reshape(-1, 1)
log_ranks = np.log(shared['ranks']).reshape(-1, 1)
features = np.column_stack((log_durations, log_ranks))
features_scaled = scaler_X.transform(features)
final_model.eval()
with torch.no_grad():
    feats_t = torch.from_numpy(features_scaled.astype(np.float32)).to(device)
    if feats_t.dim() == 2:
        feats_t_in = feats_t
    else:
        feats_t_in = feats_t
    out_scaled = final_model(feats_t_in).cpu().numpy().flatten()

# Use shared helper to inverse-transform and save IDF CSV
idf_results = build_idf_from_out_scaled(out_scaled, scaler_y, durations, frequency_factors, return_periods, standard_durations_minutes, 'idf_curves_ANN.csv')

if not val_df_combined.empty:
    # compute preds for validation (preserve original logic)
    X_val_full = val_df_combined[['log_duration', 'log_weibull_rank']].values
    X_val_scaled_full = scaler_X.transform(X_val_full)
    final_model.eval()
    with torch.no_grad():
        tX = torch.from_numpy(X_val_scaled_full.astype(np.float32)).to(device)
        out_scaled_val = final_model(tX).cpu().numpy().flatten()
    preds_log = scaler_y.inverse_transform(out_scaled_val.reshape(-1,1)).flatten()
    preds_intensity = np.exp(preds_log)
else:
    preds_intensity = np.array([])

duration_metrics, overall_metrics = compute_and_save_duration_metrics(val_df_combined, preds_intensity, duration_minutes, col_names, 'ANN', scaler_y=scaler_y)

# Store overall metrics for later use
overall_rmse_val, overall_mae_val, overall_r2_val, overall_nse_val = overall_metrics

# Expose variables expected later in the file
standard_idf_curves = idf_results['standard_idf_curves']
gumbel_idf = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "results", "idf_data.csv"))
literature_idf = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "results", "idf_lit.csv"))

# Compute per-duration validation metrics using the trained ANN and the year-based validation split
ann_duration_metrics = {}
if not val_df_combined.empty:
    # Prepare features from val_df_combined
    X_val_full = val_df_combined[['log_duration', 'log_weibull_rank']].values
    X_val_scaled_full = scaler_X.transform(X_val_full)

    # Predict with final_model
    final_model.eval()
    with torch.no_grad():
        tX = torch.from_numpy(X_val_scaled_full.astype(np.float32)).to(device)
        out_scaled = final_model(tX).cpu().numpy().flatten()

    preds_log = scaler_y.inverse_transform(out_scaled.reshape(-1,1)).flatten()
    preds_intensity = np.exp(preds_log)
    obs_intensity = val_df_combined['intensity'].values

    # Attach predictions to val_df_combined for per-duration slicing
    val_df_combined = val_df_combined.copy()
    val_df_combined['pred_intensity'] = preds_intensity

    for dmin, col in zip(duration_minutes, col_names):
        mask = (val_df_combined['duration'] == dmin)
        obs = val_df_combined.loc[mask, 'intensity'].values
        preds = val_df_combined.loc[mask, 'pred_intensity'].values
        if len(obs) == 0:
            ann_duration_metrics[col] = {'R2': np.nan, 'NSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan}
            continue
        try:
            r2m = r2_score(obs, preds)
        except Exception:
            r2m = np.nan
        nse_m = nash_sutcliffe_efficiency(obs, preds)
        mae_m = mean_absolute_error(obs, preds)
        rmse_m = np.sqrt(mean_squared_error(obs, preds))
        ann_duration_metrics[col] = {'R2': r2m, 'NSE': nse_m, 'MAE': mae_m, 'RMSE': rmse_m}
else:
    for col in col_names:
        ann_duration_metrics[col] = {'R2': np.nan, 'NSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan}

# Print concise ANN validation metrics summary
ann_avg_metrics = pd.DataFrame(ann_duration_metrics).T.mean()
print("\nANN Validation Metrics (2019-2025) by Duration:")
print(pd.DataFrame(ann_duration_metrics).T.round(4))
print("\nANN Average Metrics Across All Durations:")
print(ann_avg_metrics.round(4))

# Perform uncertainty analysis
if not val_df_combined.empty and len(preds_intensity) > 0:
    print("\nPerforming uncertainty analysis for ANN...")
    try:
        uncertainty_metrics = analyze_ai_model_uncertainty(
            model_name="ANN",
            predictions=preds_intensity,
            observations=obs_intensity
        )
        print("Uncertainty Analysis Results:")
        for key, value in uncertainty_metrics.items():
            if isinstance(value, (int, float, np.number)):
                print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"Warning: Uncertainty analysis failed: {e}")

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
    12: "1440 mins"
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

# Display metrics (comparison vs Gumbel per return period)
comparison_metrics_df = pd.DataFrame({
    'Return Period': return_periods,
    'RMSE': [round(x, 4) for x in rmse_values],
    'MAE': [round(x, 4) for x in mae_values],
    'R2': [round(x, 4) for x in r2_values],
    'NSE': [round(x, 4) for x in nse_values]
})
print("\nModel Performance Metrics by Return Period:")
print(comparison_metrics_df)

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
    1440: "1440 mins"
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
        if pd.notna(lit_value) and lit_value != '':
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
valid_lit_metrics = [(rp, rmse, mae, r2, nse) for rp, rmse, mae, r2, nse in 
                     zip(return_periods, lit_rmse_values, lit_mae_values, lit_r2_values, lit_nse_values)
                     if not np.isnan(rmse)]

if valid_lit_metrics:
    lit_metrics_df = pd.DataFrame({
        'Return Period': [x[0] for x in valid_lit_metrics],
        'RMSE': [round(x[1], 4) for x in valid_lit_metrics],
        'MAE': [round(x[2], 4) for x in valid_lit_metrics],
        'R2': [round(x[3], 4) for x in valid_lit_metrics],
        'NSE': [round(x[4], 4) for x in valid_lit_metrics]
    })
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
        'Model': 'ANN',
        'RMSE': overall_lit_rmse,
        'MAE': overall_lit_mae,
        'R2': overall_lit_r2,
        'NSE': overall_lit_nse
    }
    
    # Check if literature performance metrics file exists
    lit_metrics_file = os.path.join(os.path.dirname(__file__), "..", "results", "literature_performance_metrics.csv")
    if os.path.exists(lit_metrics_file):
        # Load existing metrics file and append/update
        lit_metrics_df = pd.read_csv(lit_metrics_file)
        
        # Check if ANN already exists in the dataframe
        if 'ANN' in lit_metrics_df['Model'].values:
            # Update existing row
            for col, val in lit_comparison_metrics.items():
                lit_metrics_df.loc[lit_metrics_df['Model'] == 'ANN', col] = val
        else:
            # Append new row
            lit_metrics_df = pd.concat([lit_metrics_df, pd.DataFrame([lit_comparison_metrics])], ignore_index=True)
    else:
        # Create new metrics dataframe
        lit_metrics_df = pd.DataFrame([lit_comparison_metrics])
    
    # Save metrics to CSV
    lit_metrics_df.to_csv(lit_metrics_file, index=False)
    print(f"Literature comparison metrics saved to: {lit_metrics_file}")

# Check if performance metrics file exists, append if it does, create if not
metrics_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "model_performance_metrics.csv")
# Use model evaluation metrics (from test set) rather than the overall comparison metrics
# rmse, mae, r2_model and nse are calculated earlier from the model evaluation on test data
metrics_row = {
    'Model': 'ANN',
    'RMSE': rmse,
    'MAE': mae,
    'R2': r2_model,
    'NSE': nse
}

if os.path.exists(metrics_file_path):
    # Load existing metrics file and append new row
    perf_df = pd.read_csv(metrics_file_path)
    
    # Check if ANN model already exists in the dataframe
    if 'ANN' in perf_df['Model'].values:
        # Update existing row
        for col, val in metrics_row.items():
            perf_df.loc[perf_df['Model'] == 'ANN', col] = val
    else:
        # Append new row
        perf_df = pd.concat([perf_df, pd.DataFrame([metrics_row])], ignore_index=True)
else:
    # Create new metrics dataframe
    perf_df = pd.DataFrame([metrics_row])

# Save metrics to CSV
perf_df.to_csv(metrics_file_path, index=False)
print(f"Model performance metrics saved to: {metrics_file_path}")

# Centralized plotting call (verbatim logic now in shared_io.plot_idf_comparisons)
# Use the raw validation metrics from training/evaluation (rmse, mae, r2_model, nse)
# when available; otherwise fall back to the comparison-derived aggregates.
try:
    plot_overall_metrics = (rmse, mae, r2_model, nse)
except NameError:
    plot_overall_metrics = (overall_rmse, overall_mae, overall_r2, overall_nse)

try:
    overall_lit_metrics = (overall_lit_rmse, overall_lit_mae, overall_lit_r2, overall_lit_nse)
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
    'ANN',
    'ann'
)

# Plot predictions vs observations after IDF comparisons
if not val_df_combined.empty and len(preds_intensity) > 0:
    y_val_intensity = val_df_combined['intensity'].values
    overall_rmse, overall_mae, overall_r2, overall_nse = plot_overall_metrics
    metrics = {
        'rmse': overall_rmse,
        'mae': overall_mae,
        'r2': overall_r2,
        'nse': overall_nse
    }
    plot_predictions_vs_observations(y_val_intensity, preds_intensity, 'ANN', 'ann', metrics)