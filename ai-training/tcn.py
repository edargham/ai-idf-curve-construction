import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared_io import (
    build_idf_from_out_scaled,
    compute_and_save_duration_metrics,
    plot_idf_comparisons,
    plot_predictions_vs_observations,
)
from shared_dataprep_tcn import create_feats_and_labels, TensorRegressionDataset
from uncertainty_analysis import analyze_ai_model_uncertainty
from shared_optuna_tuning import (
    create_tcn_objective,
    run_optuna_study,
    save_optuna_results,
    set_seed,
    get_device,
    TCN,
    train_pytorch_epoch,
    evaluate_pytorch_model
)


# Reproducibility: set a global seed and stabilize RNGs
SEED = 42
set_seed(SEED)

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

# Device selection
device = get_device()
print(f"Using device: {device}")

# Run Optuna hyperparameter tuning
print('\nStarting Optuna hyperparameter tuning for TCN...')
tcn_objective = create_tcn_objective(
    X_train_scaled, y_train_scaled,
    X_val_scaled, y_val_scaled,
    scaler_y, device, max_epochs=400
)
study = run_optuna_study(tcn_objective, "TCN", n_trials=100, direction="maximize")

# Save Optuna results
best_params_df, trials_df = save_optuna_results(study, "TCN")

# Get best hyperparameters
best_params = study.best_params

# Train final model with best hyperparameters
print("\nTraining final TCN model with best hyperparameters...")
final_model = TCN(
    seq_len=X_train_scaled.shape[1],
    num_filters=best_params['num_filters'],
    kernel_size=best_params['kernel_size'],
    dropout=best_params['dropout'],
    num_levels=best_params['num_levels']
).to(device)

optimizer = torch.optim.AdamW(
    final_model.parameters(),
    lr=best_params['lr'],
    weight_decay=best_params['weight_decay']
)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=1e-7
)

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
patience = 100
patience_counter = 0
num_epochs = 600

for epoch in range(1, num_epochs + 1):
    train_loss = train_pytorch_epoch(final_model, train_loader, optimizer, criterion, device)
    scheduler.step()
    
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

print("\nFinal TCN Validation Metrics:")
print(f"  NSE: {nse:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE: {mae:.6f}")
print(f"  R²: {r2_model:.6f}")

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

# After training, generate IDF curves using the trained PyTorch model
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
    "idf_curves_TCN.csv",
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

tcn_duration_metrics = {}
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
    tcn_duration_metrics, overall_metrics = compute_and_save_duration_metrics(
        val_df_combined,
        preds_intensity,
        duration_minutes,
        col_names,
        "TCN",
    )
    
    # Store overall metrics for later use
    overall_rmse_val, overall_mae_val, overall_r2_val, overall_nse_val = overall_metrics
else:
    for col in col_names:
        tcn_duration_metrics[col] = {
            "R2": np.nan,
            "NSE": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
        }

# Print concise TCN validation metrics summary (keeps original print format)
tcn_avg_metrics = pd.DataFrame(tcn_duration_metrics).T.mean()
print("\nTCN Validation Metrics (2019-2025) by Duration:")
print(pd.DataFrame(tcn_duration_metrics).T.round(4))
print("\nTCN Average Metrics Across All Durations:")
print(tcn_avg_metrics.round(4))

# Perform uncertainty analysis
if not val_df_combined.empty and 'preds_intensity' in locals():
    print("\nPerforming uncertainty analysis for TCN...")
    try:
        obs_intensity = val_df_combined['intensity'].values
        # Prepare validation tensor for MC Dropout
        X_val_tensor = torch.from_numpy(X_val_scaled_full.astype(np.float32)).to(device)
        
        uncertainty_metrics = analyze_ai_model_uncertainty(
            model_name="TCN",
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
metrics_file_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "model_performance_metrics.csv"
)
# Use model evaluation metrics (from test set) rather than the overall comparison metrics
# rmse, mae, r2_model and nse are calculated earlier from the model evaluation on test data
metrics_row = {"Model": "TCN", "RMSE": rmse, "MAE": mae, "R2": r2_model, "NSE": nse}

if os.path.exists(metrics_file_path):
    # Load existing metrics file and append new row
    metrics_df = pd.read_csv(metrics_file_path)

    # Check if TCN model already exists in the dataframe
    if "TCN" in metrics_df["Model"].values:
        # Update existing row
        for col, val in metrics_row.items():
            metrics_df.loc[metrics_df["Model"] == "TCN", col] = val
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


# Plot IDF curves with validation metrics
plot_idf_comparisons(
    standard_idf_curves,
    standard_durations_minutes,
    return_periods,
    (rmse, mae, r2_model, nse),
    "TCN",
    "tcn",
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

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2_model,
        'nse': nse
    }
    plot_predictions_vs_observations(y_val_intensity, preds_intensity, 'TCN', 'tcn', metrics)
