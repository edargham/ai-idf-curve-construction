import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# plotting and curve fitting are centralized in shared_io.plot_idf_comparisons
from split_utils import build_train_val
from shared_io import shared_preprocessing, build_idf_from_out_scaled, plot_idf_comparisons, plot_predictions_vs_observations
from shared_metrics import nash_sutcliffe_efficiency, squared_pearson_r2 as r2_score
from uncertainty_analysis import analyze_ai_model_uncertainty
from shared_optuna_tuning import (
    create_svm_objective,
    run_optuna_study,
    save_optuna_results,
    set_seed
)

# Set reproducibility seed
SEED = 42
set_seed(SEED)


# Use centralized preprocessing (verbatim copy lives in shared_io.shared_preprocessing)
shared = shared_preprocessing()
# expose variables expected by the rest of the script without changing logic
df = shared['df']
combined_df = shared['combined_df']
duration_minutes = shared['duration_minutes']
col_names = shared['col_names']
# Ensure the canonical year-based train/validation split is created here (verbatim behavior)
train_df_combined, val_df_combined, years = build_train_val(df)

# Transform (log) for modeling
train_df_combined['log_duration'] = np.log(train_df_combined['duration'])
train_df_combined['log_weibull_rank'] = np.log(train_df_combined['weibull_rank'])
train_df_combined['log_intensity'] = np.log(train_df_combined['intensity'])

if not val_df_combined.empty:
    val_df_combined['log_duration'] = np.log(val_df_combined['duration'])
    val_df_combined['log_weibull_rank'] = np.log(val_df_combined['weibull_rank'])
    val_df_combined['log_intensity'] = np.log(val_df_combined['intensity'])

# Prepare X/y
X_train = train_df_combined[['log_duration','log_weibull_rank']]
y_train = train_df_combined['log_intensity']

X_val = val_df_combined[['log_duration','log_weibull_rank']] if not val_df_combined.empty else None
y_val = val_df_combined['log_intensity'] if not val_df_combined.empty else None

# Standard scale the data (fit on train only)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val) if X_val is not None and len(X_val)>0 else None

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1)).flatten()
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1,1)).flatten() if y_val is not None and len(y_val)>0 else None

# Run Optuna hyperparameter tuning
print('Starting Optuna hyperparameter tuning on training/validation split...')
svm_objective = create_svm_objective(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler_y)
study = run_optuna_study(svm_objective, "SVM", n_trials=100, direction="maximize")

# Save Optuna results
best_params_df, trials_df = save_optuna_results(study, "SVM")

# Get best hyperparameters
best_params = study.best_params

# Train final model with best hyperparameters on full training set
print("\nTraining final SVM model with best hyperparameters...")
model = SVR(**best_params)
model.fit(X_train_scaled, y_train_scaled)

# Predict on validation split (2019-2025)
if X_val_scaled is not None and len(X_val_scaled)>0:
    y_pred_scaled = model.predict(X_val_scaled)
    y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    # Model was trained on log(intensity), but earlier code exponentiated after inverse transform, so keep same convention
    y_pred_intensity = np.exp(y_pred_log)
    y_val_intensity = np.exp(y_val.values)
else:
    y_pred_intensity = np.array([])
    y_val_intensity = np.array([])

# Compute per-duration metrics on validation split
svm_duration_metrics = {}
for dmin, col in zip(duration_minutes, col_names):
    # select rows for this duration in val_df_combined
    if val_df_combined.empty:
        svm_duration_metrics[col] = {'R2': np.nan, 'NSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan}
        continue

    mask = (val_df_combined['duration'] == dmin)
    obs = val_df_combined.loc[mask, 'intensity'].values
    if len(obs) == 0:
        svm_duration_metrics[col] = {'R2': np.nan, 'NSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan}
        continue

    preds = y_pred_intensity[mask.values]
    # Safety: if sizes mismatch, try to align by positions
    if len(preds) != len(obs):
        # attempt to slice sequentially by counting rows per duration
        # build mapping of indices
        idxs = val_df_combined.index[val_df_combined['duration']==dmin].tolist()
        preds = np.array([y_pred_intensity[i] for i in range(len(y_pred_intensity)) if i in idxs])

    try:
        r2m = r2_score(obs, preds)
    except Exception:
        r2m = np.nan
    nse_m = nash_sutcliffe_efficiency(obs, preds)
    mae_m = mean_absolute_error(obs, preds)
    rmse_m = np.sqrt(mean_squared_error(obs, preds))

    svm_duration_metrics[col] = {'R2': r2m, 'NSE': nse_m, 'MAE': mae_m, 'RMSE': rmse_m}

# Compute overall validation metrics (global) from the validation predictions
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

# Plot predictions vs observations
if len(y_val_intensity) > 0 and len(y_pred_intensity) == len(y_val_intensity):
    metrics = {'rmse': rmse, 'mae': mae, 'r2': r2_model, 'nse': nse}
    plot_predictions_vs_observations(y_val_intensity, y_pred_intensity, 'SVM', 'svm', metrics)
    
    # Perform uncertainty analysis
    print("\nPerforming uncertainty analysis for SVM...")
    try:
        uncertainty_metrics = analyze_ai_model_uncertainty(
            model_name="SVM",
            predictions=y_pred_intensity,
            observations=y_val_intensity,
            model=model,
            X_val=X_val_scaled,
            scaler_y=scaler_y,
            use_bootstrap=True,
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            n_bootstrap=50
        )
        print("Uncertainty Analysis Results:")
        for key, value in uncertainty_metrics.items():
            if isinstance(value, (int, float, np.number)):
                print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"Warning: Uncertainty analysis failed: {e}")

# Use shared preprocessing constants and helpers
durations = shared['durations']
frequency_factors = shared['frequency_factors']
return_periods = shared['return_periods']
standard_durations_minutes = shared['standard_durations_minutes']

# Predict IDF using model (base log intensities), then build/save CSV via shared helper
log_durations = np.log(durations).reshape(-1, 1)
log_ranks = np.log(shared['ranks']).reshape(-1, 1)
features = np.column_stack((log_durations, log_ranks))
features_scaled = scaler_X.transform(features)
base_log_intensities_scaled = model.predict(features_scaled)
# Use shared helper to inverse transform and save CSV exactly as before
idf_results = build_idf_from_out_scaled(base_log_intensities_scaled, scaler_y, durations, frequency_factors, return_periods, standard_durations_minutes, 'idf_curves_SVM.csv')

# Expose variables expected later in the original script
standard_idf_curves = idf_results['standard_idf_curves']

# SVM validation metrics have been computed and saved above in
# `validation_csv_path` (SVM columns). Print a concise summary here.
svm_avg_metrics = pd.DataFrame(svm_duration_metrics).T.mean()
print("\nSVM Validation Metrics (2019-2025) by Duration:")
print(pd.DataFrame(svm_duration_metrics).T.round(4))
print("\nSVM Average Metrics Across All Durations:")
print(svm_avg_metrics.round(4))

# Check if performance metrics file exists, append if it does, create if not
metrics_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "model_performance_metrics.csv")
# Use model evaluation metrics (from test set) rather than the overall comparison metrics
# rmse, mae, r2_model and nse are calculated earlier from the model evaluation on test data
metrics_row = {
    'Model': 'SVM',
    'RMSE': rmse,
    'MAE': mae,
    'R2': r2_model,
    'NSE': nse
}

if os.path.exists(metrics_file_path):
    # Load existing metrics file and append new row
    perf_df = pd.read_csv(metrics_file_path)
    
    # Check if SVM model already exists in the dataframe
    if 'SVM' in perf_df['Model'].values:
        # Update existing row
        for col, val in metrics_row.items():
            perf_df.loc[perf_df['Model'] == 'SVM', col] = val
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
    'SVM',
    'svm'
)
