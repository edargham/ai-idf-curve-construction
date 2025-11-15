import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Additional imports required by the ANN script
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from split_utils import build_train_val
from shared_io import shared_preprocessing, build_idf_from_out_scaled, compute_and_save_duration_metrics, plot_idf_comparisons, plot_predictions_vs_observations
from shared_metrics import nash_sutcliffe_efficiency, squared_pearson_r2 as r2_score
from uncertainty_analysis import analyze_ai_model_uncertainty
from shared_dataprep_tcn import TensorRegressionDataset
from shared_optuna_tuning import (
    create_ann_objective,
    run_optuna_study,
    save_optuna_results,
    set_seed,
    get_device,
    MLP,
    train_pytorch_epoch,
    evaluate_pytorch_model
)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Limit CUDA workspace for stability
torch.use_deterministic_algorithms(True)  # Enforce deterministic algorithms in PyTorch
# Reproducibility: set a global seed and stabilize RNGs
SEED = 42
set_seed(SEED)

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

# Device selection
device = get_device()
print(f"Using device: {device}")

# Run Optuna hyperparameter tuning
print('\nStarting Optuna hyperparameter tuning for ANN...')
ann_objective = create_ann_objective(
    X_train_scaled, y_train_scaled, 
    X_val_scaled, y_val_scaled,
    scaler_y, device, max_epochs=300
)
study = run_optuna_study(ann_objective, "ANN", n_trials=100, direction="maximize")

# Save Optuna results
best_params_df, trials_df = save_optuna_results(study, "ANN")

# Get best hyperparameters
best_params = study.best_params

# Train final model with best hyperparameters
print("\nTraining final ANN model with best hyperparameters...")
final_model = MLP(
    input_dim=X_train_scaled.shape[1],
    hidden_size=best_params['hidden_size'],
    dropout=best_params['dropout']
).to(device)

optimizer = torch.optim.AdamW(
    final_model.parameters(),
    lr=best_params['lr'],
    weight_decay=best_params['weight_decay']
)
criterion = nn.MSELoss()

# Create reproducible data loader
train_loader_gen = torch.Generator()
train_loader_gen.manual_seed(SEED)

train_ds = TensorRegressionDataset(X_train_scaled, y_train_scaled)
train_loader = DataLoader(
    train_ds, 
    batch_size=best_params['batch_size'], 
    shuffle=True, 
    generator=train_loader_gen
)

val_ds = TensorRegressionDataset(X_val_scaled, y_val_scaled)
val_loader = DataLoader(val_ds, batch_size=best_params['batch_size'], shuffle=False)

# Training loop with early stopping
best_val_nse = -np.inf
best_weights = None
patience = 50
patience_counter = 0
num_epochs = 500

for epoch in range(1, num_epochs + 1):
    train_loss = train_pytorch_epoch(final_model, train_loader, optimizer, criterion, device)
    
    # Evaluate on validation (with inverse transform to get original scale metrics)
    val_metrics = evaluate_pytorch_model(final_model, val_loader, device, scaler_y=scaler_y)
    val_nse = val_metrics['nse']
    
    # Early stopping
    if val_nse > best_val_nse:
        best_val_nse = val_nse
        best_weights = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val NSE={val_nse:.6f}")

# Restore best weights
if best_weights is not None:
    final_model.load_state_dict(best_weights)
    print(f"\nRestored best model weights (NSE={best_val_nse:.6f})")

# Evaluate final model (with inverse transform to get original scale metrics)
final_metrics = evaluate_pytorch_model(final_model, val_loader, device, scaler_y=scaler_y)
rmse = final_metrics['rmse']
mae = final_metrics['mae']
r2_model = final_metrics['r2']
nse = final_metrics['nse']

print("\nFinal ANN Validation Metrics:")
print(f"  NSE: {nse:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE: {mae:.6f}")
print(f"  R²: {r2_model:.6f}")

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
        # Prepare validation tensor for MC Dropout
        X_val_tensor = torch.from_numpy(X_val_scaled_full.astype(np.float32)).to(device)
        
        uncertainty_metrics = analyze_ai_model_uncertainty(
            model_name="ANN",
            predictions=preds_intensity,
            observations=obs_intensity,
            model=final_model,
            X_val=X_val_tensor,
            device=device,
            scaler_y=scaler_y,
            use_mc_dropout=True,
            n_mc_samples=50
        )
        print("Uncertainty Analysis Results:")
        for key, value in uncertainty_metrics.items():
            if isinstance(value, (int, float, np.number)):
                print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"Warning: Uncertainty analysis failed: {e}")

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

# Plot IDF curves with validation metrics
plot_idf_comparisons(
    standard_idf_curves,
    standard_durations_minutes,
    return_periods,
    (rmse, mae, r2_model, nse),
    'ANN',
    'ann'
)

# Plot predictions vs observations after IDF comparisons
if not val_df_combined.empty and len(preds_intensity) > 0:
    y_val_intensity = val_df_combined['intensity'].values
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2_model,
        'nse': nse
    }
    plot_predictions_vs_observations(y_val_intensity, preds_intensity, 'ANN', 'ann', metrics)
