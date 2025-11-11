import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# plotting and curve fitting are centralized in shared_io.plot_idf_comparisons
from split_utils import build_train_val
from shared_io import shared_preprocessing, build_idf_from_out_scaled, plot_idf_comparisons, plot_predictions_vs_observations
from shared_metrics import nash_sutcliffe_efficiency, squared_pearson_r2 as r2_score
from uncertainty_analysis import analyze_ai_model_uncertainty


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

grid_search_params = {
    'C': [0.1, 1.0, 10.0],
    'epsilon': [0.01, 0.1, 0.2],
    'gamma': [0.01, 0.1, 1.0],
    'kernel': ['rbf']
}

print('Starting hyperparameter tuning on training split (1998-2018)...')
tuner = GridSearchCV(SVR(), grid_search_params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)
tuner.fit(X_train_scaled, y_train_scaled)

print('\nBest parameters from grid search:', tuner.best_params_)
print('Best cross-validation MSE score:', -tuner.best_score_)

# Fit final model on full training set
model = SVR(**(tuner.best_params_))
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
            observations=y_val_intensity
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
gumbel_idf = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "results", "idf_data.csv"))
literature_idf = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "results", "idf_lit.csv"))

# SVM validation metrics have been computed and saved above in
# `validation_csv_path` (SVM columns). Print a concise summary here.
svm_avg_metrics = pd.DataFrame(svm_duration_metrics).T.mean()
print("\nSVM Validation Metrics (2019-2025) by Duration:")
print(pd.DataFrame(svm_duration_metrics).T.round(4))
print("\nSVM Average Metrics Across All Durations:")
print(svm_avg_metrics.round(4))

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
        'Model': 'SVM',
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
        
        # Check if SVM already exists in the dataframe
        if 'SVM' in lit_metrics_df['Model'].values:
            # Update existing row
            for col, val in lit_comparison_metrics.items():
                lit_metrics_df.loc[lit_metrics_df['Model'] == 'SVM', col] = val
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

# Centralized plotting: prefer the raw validation metrics (from model evaluation)
# for the model overlay. If those aren't available, fall back to the IDF-comparison
# aggregate metrics (overall_rmse, etc.) which were computed earlier.
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
    'SVM',
    'svm'
)
