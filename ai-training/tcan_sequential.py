import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared_io import (
    build_idf_from_direct_predictions,
    compute_sequential_metrics,
    plot_idf_comparisons,
    plot_predictions_vs_observations,
)
from shared_dataprep_tcn import SequentialIDFDataset
from uncertainty_analysis import analyze_ai_model_uncertainty
from shared_optuna_tuning import (
    create_sequential_tcan_objective,
    run_optuna_study,
    save_optuna_results,
    set_seed,
    get_device,
    SequentialTCAN,
    train_sequential_epoch,
    evaluate_sequential_model
)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

# Reproducibility
SEED = 42
set_seed(SEED)

# Paths
csv_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "bey-aggregated-final.csv"
)

# Device selection
device = get_device()
print(f"Using device: {device}")

# Create datasets with initial hyperparameters
print("\n" + "="*80)
print("CREATING SEQUENTIAL IDF DATASETS FOR TCAN")
print("="*80)

# Initial hyperparameters for dataset creation (will be tuned by Optuna)
initial_seq_len = 256
initial_extreme_percentile = 75
initial_extreme_ratio = 0.8

train_dataset = SequentialIDFDataset(
    csv_path=csv_path,
    train_years=range(1998, 2019),
    val_years=range(2019, 2026),
    seq_len=initial_seq_len,
    extreme_percentile=initial_extreme_percentile,
    extreme_ratio=initial_extreme_ratio,
    is_train=True,
    seed=SEED
)

val_dataset = SequentialIDFDataset(
    csv_path=csv_path,
    train_years=range(1998, 2019),
    val_years=range(2019, 2026),
    seq_len=initial_seq_len,
    extreme_percentile=initial_extreme_percentile,
    extreme_ratio=initial_extreme_ratio,
    is_train=False,
    seed=SEED
)

# Run Optuna hyperparameter tuning
print('\n' + "="*80)
print('STARTING OPTUNA HYPERPARAMETER TUNING FOR SEQUENTIAL TCAN')
print("="*80)

tcan_objective = create_sequential_tcan_objective(
    train_dataset, val_dataset, device, max_epochs=200
)
study = run_optuna_study(tcan_objective, "TCAN_Sequential", n_trials=1, direction="maximize")

# Save Optuna results
best_params_df, trials_df = save_optuna_results(study, "TCAN_Sequential")

# Get best hyperparameters
best_params = study.best_params
print(f"\nBest Hyperparameters (NSE = {study.best_value:.6f}):")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# Recreate datasets with best hyperparameters
print("\n" + "="*80)
print("RECREATING DATASETS WITH BEST HYPERPARAMETERS")
print("="*80)

train_dataset_final = SequentialIDFDataset(
    csv_path=csv_path,
    train_years=range(1998, 2019),
    val_years=range(2019, 2026),
    seq_len=best_params['seq_len'],
    extreme_percentile=best_params['extreme_percentile'],
    extreme_ratio=best_params['extreme_ratio'],
    is_train=True,
    seed=SEED
)

val_dataset_final = SequentialIDFDataset(
    csv_path=csv_path,
    train_years=range(1998, 2019),
    val_years=range(2019, 2026),
    seq_len=best_params['seq_len'],
    extreme_percentile=best_params['extreme_percentile'],
    extreme_ratio=best_params['extreme_ratio'],
    is_train=False,
    seed=SEED
)

# Train final model with best hyperparameters
print("\n" + "="*80)
print("TRAINING FINAL TCAN MODEL WITH BEST HYPERPARAMETERS")
print("="*80)

final_model = SequentialTCAN(
    num_channels=13,
    seq_len=best_params['seq_len'],
    num_filters=best_params['num_filters'],
    kernel_size=best_params['kernel_size'],
    dropout=best_params['dropout'],
    num_levels=best_params['num_levels'],
    attn_heads=best_params['attn_heads'],
    attn_dropout=best_params['attn_dropout']
).to(device)

optimizer = torch.optim.AdamW(
    final_model.parameters(),
    lr=best_params['lr'],
    weight_decay=best_params['weight_decay']
)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=20, min_lr=1e-7
)

# Create data loaders
train_loader_gen = torch.Generator()
train_loader_gen.manual_seed(SEED)
train_loader = DataLoader(
    train_dataset_final,
    batch_size=best_params['batch_size'],
    shuffle=True,
    generator=train_loader_gen
)
val_loader = DataLoader(
    val_dataset_final,
    batch_size=best_params['batch_size'],
    shuffle=False
)

# Training loop with early stopping
best_val_nse = -np.inf
best_weights = None
patience = 80
patience_counter = 0
num_epochs = 400

print("\nTraining progress:")
for epoch in range(1, num_epochs + 1):
    train_loss = train_sequential_epoch(final_model, train_loader, optimizer, criterion, device)
    val_metrics = evaluate_sequential_model(final_model, val_loader, device, 
                                            scaler_y=train_dataset_final.scaler_y)
    val_nse = val_metrics['nse']
    
    scheduler.step(val_nse)
    
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
        print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val NSE={val_nse:.6f}")

# Restore best weights
if best_weights is not None:
    final_model.load_state_dict(best_weights)
    print(f"\nRestored best model weights (NSE={best_val_nse:.6f})")

# Evaluate final model
final_metrics = evaluate_sequential_model(final_model, val_loader, device, 
                                          scaler_y=train_dataset_final.scaler_y)
rmse = final_metrics['rmse']
mae = final_metrics['mae']
r2_model = final_metrics['r2']
nse = final_metrics['nse']

print("\n" + "="*80)
print("FINAL TCAN VALIDATION METRICS")
print("="*80)
print(f"  NSE:  {nse:.6f}")
print(f"  RMSE: {rmse:.6f} mm/hr")
print(f"  MAE:  {mae:.6f} mm/hr")
print(f"  R²:   {r2_model:.6f}")

# Generate IDF curves using the trained model
print("\n" + "="*80)
print("GENERATING IDF CURVES")
print("="*80)

final_model.eval()
with torch.no_grad():
    # Get the last validation window for prediction
    last_window_scaled, last_target_scaled = val_dataset_final.get_latest_window()
    if last_window_scaled is not None:
        last_window_tensor = torch.from_numpy(last_window_scaled).to(device)  # [1, 13, seq_len]
        predictions_scaled = final_model(last_window_tensor)  # [1, 13, 6]
        predictions_scaled = predictions_scaled.cpu().numpy()[0]  # [13, 6]
    else:
        # Fallback: use mean of all validation predictions
        all_preds = []
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = final_model(xb)
            all_preds.append(out.cpu())
        predictions_scaled = torch.cat(all_preds).mean(dim=0).numpy()  # [13, 6]

# Build IDF curves
standard_durations_minutes = [5, 10, 15, 30, 60, 90, 120, 180, 360, 720, 900, 1080, 1440]
return_periods = [2, 5, 10, 25, 50, 100]

idf_artifacts = build_idf_from_direct_predictions(
    predictions_scaled,
    train_dataset_final.scaler_y,
    standard_durations_minutes,
    return_periods,
    "idf_curves_TCAN.csv",
    duration_stats=train_dataset_final.duration_stats
)

standard_idf_curves = idf_artifacts["standard_idf_curves"]
idf_curves = idf_artifacts["idf_curves"]

# Compute detailed metrics using validation data
print("\n" + "="*80)
print("COMPUTING DETAILED VALIDATION METRICS")
print("="*80)

# Get all validation predictions and targets
all_preds_scaled = []
all_targets_scaled = []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        out = final_model(xb)
        all_preds_scaled.append(out.cpu().numpy())
        all_targets_scaled.append(yb.numpy())

all_preds_scaled = np.concatenate(all_preds_scaled, axis=0)  # [N, 13, 6]
all_targets_scaled = np.concatenate(all_targets_scaled, axis=0)  # [N, 13, 6]

# Average across all samples for final IDF
mean_preds_scaled = all_preds_scaled.mean(axis=0)  # [13, 6]
mean_targets_scaled = all_targets_scaled.mean(axis=0)  # [13, 6]

# Compute metrics
tcan_duration_metrics, overall_metrics = compute_sequential_metrics(
    mean_preds_scaled,
    mean_targets_scaled,
    train_dataset_final.scaler_y,
    standard_durations_minutes,
    return_periods,
    "TCAN"
)

overall_rmse_val, overall_mae_val, overall_r2_val, overall_nse_val = overall_metrics

# Print concise metrics summary
tcan_avg_metrics = pd.DataFrame(tcan_duration_metrics).T.mean()
print("\nTCAN Validation Metrics by Duration:")
print(pd.DataFrame(tcan_duration_metrics).T.round(4))
print("\nTCAN Average Metrics Across All Durations:")
print(tcan_avg_metrics.round(4))

# Perform uncertainty analysis
print("\n" + "="*80)
print("UNCERTAINTY ANALYSIS")
print("="*80)

try:
    # Prepare data for uncertainty analysis
    preds_unscaled = train_dataset_final.scaler_y.inverse_transform(
        all_preds_scaled.reshape(-1, 1)
    ).reshape(all_preds_scaled.shape)
    targets_unscaled = train_dataset_final.scaler_y.inverse_transform(
        all_targets_scaled.reshape(-1, 1)
    ).reshape(all_targets_scaled.shape)
    
    # Flatten for uncertainty analysis
    preds_flat = preds_unscaled.flatten()
    obs_flat = targets_unscaled.flatten()
    
    # Get validation tensor for MC Dropout
    val_sequences = []
    for xb, _ in val_loader:
        val_sequences.append(xb)
    X_val_tensor = torch.cat(val_sequences).to(device)
    
    uncertainty_metrics = analyze_ai_model_uncertainty(
        model_name="TCAN",
        predictions=preds_flat,
        observations=obs_flat,
        model=final_model,
        X_val=X_val_tensor,
        device=device,
        scaler_y=train_dataset_final.scaler_y,
        use_mc_dropout=True,
        n_mc_samples=50
    )
    
    print("Uncertainty Analysis Results:")
    for key, value in uncertainty_metrics.items():
        if isinstance(value, (int, float, np.number)):
            print(f"  {key}: {value:.4f}")
except Exception as e:
    print(f"Warning: Uncertainty analysis failed: {e}")

# Plot IDF curves
print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

plot_idf_comparisons(
    standard_idf_curves,
    standard_durations_minutes,
    return_periods,
    (rmse, mae, r2_model, nse),
    "TCAN (Sequential)",
    "tcan_sequential",
)

# Plot predictions vs observations
try:
    # Use flattened predictions for scatter plot
    plot_predictions_vs_observations(
        obs_flat,
        preds_flat,
        'TCAN (Sequential)',
        'tcan_sequential',
        {'rmse': rmse, 'mae': mae, 'r2': r2_model, 'nse': nse}
    )
except Exception as e:
    print(f"Warning: Could not create predictions vs observations plot: {e}")

print("\n" + "="*80)
print("TCAN TRAINING AND EVALUATION COMPLETE")
print("="*80)
