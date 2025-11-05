import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directory to path to import shared modules
sys.path.insert(0, os.path.dirname(__file__))

from shared_preprocessing import (
    load_annual_max_intensity,
    split_train_validation,
    get_duration_configs,
    get_return_periods,
)
from shared_postprocessing import (
    save_performance_metrics,
    create_idf_plot,
    plot_model_vs_literature_comparison,
    plot_literature_only_idf,
    compare_with_literature,
    get_literature_duration_mapping,
    r2_score,
    nash_sutcliffe_efficiency,
)


def calculate_empirical_return_periods(data):
    """
    Calculate empirical return periods using Weibull plotting position:
    T = (n+1)/m where n is sample size and m is rank (1 = largest).
    Returns sorted data (desc) and corresponding return periods.
    """
    n = len(data)
    if n == 0:
        return np.array([]), np.array([])
    sorted_data = np.sort(data)[::-1]  # Sort in descending order
    ranks = np.arange(1, n + 1)
    return_periods = (n + 1) / ranks
    return sorted_data, return_periods


def calculate_validation_metrics(val_df, durations, model_params, model_type="lognormal"):
    """Calculate validation metrics for a given model across all durations."""
    duration_metrics = {}

    for j, dur in enumerate(durations):
        val_values = val_df[dur].dropna().values
        if len(val_values) == 0:
            duration_metrics[dur] = {
                "R2": np.nan,
                "NSE": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
            }
            continue

        sorted_obs, empirical_T = calculate_empirical_return_periods(val_values)

        if model_type == "lognormal":
            params = model_params.get(dur, (np.nan, np.nan))
            shape, loc, scale = params

            if np.isnan(scale) or scale <= 0:
                duration_metrics[dur] = {
                    "R2": np.nan,
                    "NSE": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                }
                continue

            # Predict intensities at empirical return periods
            qs = 1 - 1 / empirical_T
            eps = 1e-6
            qs = np.clip(qs, eps, 1 - eps)
            predicted = stats.lognorm.ppf(q=qs, s=shape, loc=loc, scale=scale)

        # Calculate metrics
        try:
            r2 = r2_score(sorted_obs, predicted)
        except Exception:
            r2 = np.nan

        nse = nash_sutcliffe_efficiency(sorted_obs, predicted)
        mae = mean_absolute_error(sorted_obs, predicted)
        rmse = np.sqrt(mean_squared_error(sorted_obs, predicted))

        duration_metrics[dur] = {"R2": r2, "NSE": nse, "MAE": mae, "RMSE": rmse}

    return duration_metrics


def fit_lognormal_distribution(train_df, durations, return_periods, probabilities):
    """
    Fit Log-Normal distribution on training data and calculate intensities.
    
    Parameters:
        train_df (pd.DataFrame): Training data with durations as columns
        durations (list): List of duration column names
        return_periods (np.array): Array of return periods
        probabilities (np.array): Array of probabilities (1 - 1/T)
        
    Returns:
        tuple: (lognormal_params, intensities_lognormal)
            - lognormal_params: Dictionary of (shape, loc, scale) for each duration
            - intensities_lognormal: 2D array of intensities [return_periods x durations]
    """
    lognormal_params = {}
    intensities_lognormal = np.zeros((len(return_periods), len(durations)))
    
    for j, dur in enumerate(durations):
        # Fit on train data for this duration
        train_values = train_df[dur].dropna().values
        if len(train_values) < 2:
            # not enough data to fit
            lognormal_params[dur] = (np.nan, np.nan, np.nan)
            intensities_lognormal[:, j] = np.nan
            continue

        # Fit Log-Normal distribution (shape, loc, scale)
        shape, loc, scale = stats.lognorm.fit(train_values, floc=0)
        lognormal_params[dur] = (shape, loc, scale)

        for i, prob in enumerate(probabilities):
            intensities_lognormal[i, j] = stats.lognorm.ppf(q=prob, s=shape, loc=loc, scale=scale)
    
    return lognormal_params, intensities_lognormal


def validate_literature_idf(val_df, lit_data, durations, duration_hours, return_periods):
    """
    Validate literature IDF curves against validation dataset.
    
    Parameters:
        val_df (pd.DataFrame): Validation data
        lit_data (pd.DataFrame): Literature IDF data
        durations (list): List of duration names
        duration_hours (list): List of durations in hours
        return_periods (np.array): Return periods
        
    Returns:
        dict: Average validation metrics for literature data
    """
    lit_duration_mapping = get_literature_duration_mapping()
    literature_duration_metrics = {}

    for j, dur in enumerate(durations):
        # Only validate durations available in literature data
        duration_min = int(duration_hours[j] * 60)
        if duration_min not in lit_duration_mapping:
            literature_duration_metrics[dur] = {
                "R2": np.nan,
                "NSE": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
            }
            continue

        # Get observed validation values for this duration
        val_values = val_df[dur].dropna().values
        if len(val_values) == 0:
            literature_duration_metrics[dur] = {
                "R2": np.nan,
                "NSE": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
            }
            continue

        sorted_obs, empirical_T = calculate_empirical_return_periods(val_values)

        # For each empirical return period, interpolate literature IDF values
        lit_col = lit_duration_mapping[duration_min]

        # Get all return periods and corresponding literature intensities for this duration
        lit_return_periods = []
        lit_intensities = []
        for rp in return_periods:
            lit_row = lit_data[lit_data["Return Period (years)"] == rp].iloc[0]
            lit_value = lit_row[lit_col]
            if not (pd.isna(lit_value) or lit_value == "" or lit_value == 0):
                lit_return_periods.append(rp)
                lit_intensities.append(float(lit_value))

        if len(lit_return_periods) < 2:
            # Not enough literature data points for interpolation
            literature_duration_metrics[dur] = {
                "R2": np.nan,
                "NSE": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
            }
            continue

        # Interpolate literature values at empirical return periods
        # Use log-log interpolation as is common for IDF relationships
        log_lit_T = np.log(lit_return_periods)
        log_lit_I = np.log(lit_intensities)

        # Clip empirical return periods to literature range to avoid extrapolation issues
        min_lit_T, max_lit_T = min(lit_return_periods), max(lit_return_periods)
        valid_mask = (empirical_T >= min_lit_T) & (empirical_T <= max_lit_T)

        if not valid_mask.any():
            literature_duration_metrics[dur] = {
                "R2": np.nan,
                "NSE": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
            }
            continue

        clipped_emp_T = empirical_T[valid_mask]
        clipped_obs = sorted_obs[valid_mask]
        log_clipped_T = np.log(clipped_emp_T)

        # Perform linear interpolation in log-log space
        predicted_log_I = np.interp(log_clipped_T, log_lit_T, log_lit_I)
        predicted = np.exp(predicted_log_I)

        # Compute metrics comparing observed vs literature-predicted values
        try:
            r2 = r2_score(clipped_obs, predicted)
        except Exception:
            r2 = np.nan

        nse = nash_sutcliffe_efficiency(clipped_obs, predicted)
        mae = mean_absolute_error(clipped_obs, predicted)
        rmse = np.sqrt(mean_squared_error(clipped_obs, predicted))

        literature_duration_metrics[dur] = {
            "R2": r2,
            "NSE": nse,
            "MAE": mae,
            "RMSE": rmse,
        }

    # Aggregate literature validation metrics
    lit_avg_metrics = pd.DataFrame(literature_duration_metrics).T.mean()
    
    return lit_avg_metrics


def main():
    """Main function to run Log-Normal distribution analysis."""
    
    print("\n" + "=" * 60)
    print("LOG-NORMAL DISTRIBUTION IDF ANALYSIS")
    print("=" * 60)
    
    # Load preprocessed data
    print("\nLoading annual maximum intensity data...")
    annual_max_intensity = load_annual_max_intensity()
    
    # Get configurations
    durations, duration_hours, duration_labels = get_duration_configs()
    return_periods, probabilities = get_return_periods()
    
    # Split into training and validation
    print("Splitting data into training (1998-2018) and validation (2019-2025)...")
    train_df, val_df = split_train_validation(annual_max_intensity)
    
    # Fit Log-Normal distribution
    print("\nFitting Log-Normal distribution on training data...")
    lognormal_params, intensities_lognormal = fit_lognormal_distribution(
        train_df, durations, return_periods, probabilities
    )
    
    print("\nLog-Normal Distribution Parameters:")
    for dur in durations:
        shape, loc, scale = lognormal_params[dur]
        print(f"{dur}: shape = {shape:.4f}, location = {loc:.4f}, scale = {scale:.4f}")
    
    # Validate on validation set
    print("\n" + "=" * 60)
    print("VALIDATION ON 2019-2025 USING LOG-NORMAL FIT FROM 1998-2018")
    print("=" * 60)
    
    duration_metrics = calculate_validation_metrics(
        val_df, durations, lognormal_params, "lognormal"
    )
    
    # Aggregate and display validation metrics
    avg_metrics = pd.DataFrame(duration_metrics).T.mean()
    print("\nIDF Validation Metrics on 2019-2025 (by Duration):")
    print(pd.DataFrame(duration_metrics).T.round(4))
    print("\nAverage Metrics Across All Durations:")
    print(avg_metrics.round(4))
    
    # Save validation metrics
    try:
        save_performance_metrics(
            "Log-Normal",
            avg_metrics.get("RMSE", np.nan),
            avg_metrics.get("MAE", np.nan),
            avg_metrics.get("R2", np.nan),
            avg_metrics.get("NSE", np.nan),
        )
    except Exception as e:
        print(
            f"Warning: failed to save Log-Normal metrics to model_performance_metrics.csv: {e}"
        )
    
    # Save IDF data
    print("\nSaving IDF curves data...")
    idf_data = pd.DataFrame(intensities_lognormal, index=return_periods, columns=duration_labels)
    idf_data.index.name = "Return Period (years)"
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    idf_data.to_csv(os.path.join(output_dir, "idf_data_lognormal.csv"))
    print(f"IDF data saved to: {os.path.join(output_dir, 'idf_data_lognormal.csv')}")
    
    # Create IDF plot
    print("\nGenerating IDF curves plot...")
    colors = ["blue", "green", "red", "purple", "orange", "brown"]
    create_idf_plot(
        intensities_lognormal,
        return_periods,
        duration_hours,
        colors,
        "Intensity-Duration-Frequency (IDF) Curves - Log-Normal",
        "idf_curves_lognormal.png",
        avg_metrics,
        show_power_law_params=True,
    )
    
    # Compare with literature if available
    print("\n" + "=" * 60)
    print("COMPARISON WITH LITERATURE DATA")
    print("=" * 60)
    
    lit_data_path = os.path.join(os.path.dirname(__file__), "..", "results", "idf_lit.csv")
    if os.path.exists(lit_data_path):
        lit_data = pd.read_csv(lit_data_path)
        lit_duration_mapping = get_literature_duration_mapping()
        
        # Compare model with literature data
        comparison_metrics = compare_with_literature(
            idf_data, lit_data, return_periods, "Log-Normal"
        )
        overall_rmse = comparison_metrics["RMSE"]
        overall_mae = comparison_metrics["MAE"]
        overall_r2 = comparison_metrics["R2"]
        overall_nse = comparison_metrics["NSE"]
        
        # Create comparison plot with metrics overlay
        comparison_plot_metrics = {
            "RMSE": overall_rmse,
            "MAE": overall_mae,
            "R2": overall_r2,
            "NSE": overall_nse,
        }
        plot_model_vs_literature_comparison(
            intensities_lognormal,
            lit_data,
            return_periods,
            duration_hours,
            lit_duration_mapping,
            "Log-Normal",
            "idf_comparison_lognormal.png",
            comparison_plot_metrics,
        )
        print(
            f"\nLiterature comparison plot saved to: "
            f"{os.path.join(os.path.dirname(__file__), '..', 'figures', 'idf_comparison_lognormal.png')}"
        )
        
        # Save comparison metrics
        comparison_metrics_row = {
            "Model": "Log-Normal",
            "RMSE": overall_rmse,
            "MAE": overall_mae,
            "R2": overall_r2,
            "NSE": overall_nse,
        }
        
        # Check if literature performance metrics file exists
        lit_metrics_file = os.path.join(
            os.path.dirname(__file__), "..", "results", "literature_performance_metrics.csv"
        )
        if os.path.exists(lit_metrics_file):
            # Load existing metrics and update/append
            existing_metrics = pd.read_csv(lit_metrics_file)
            
            # Check if Log-Normal already exists
            if "Log-Normal" in existing_metrics["Model"].values:
                # Update existing row
                for col, val in comparison_metrics_row.items():
                    existing_metrics.loc[
                        existing_metrics["Model"] == "Log-Normal", col
                    ] = val
            else:
                # Append new row
                existing_metrics = pd.concat(
                    [existing_metrics, pd.DataFrame([comparison_metrics_row])],
                    ignore_index=True,
                )
            
            existing_metrics.to_csv(lit_metrics_file, index=False)
        else:
            # Create new file
            pd.DataFrame([comparison_metrics_row]).to_csv(lit_metrics_file, index=False)
        
        print(f"Literature comparison metrics saved to: {lit_metrics_file}")
        
        # LITERATURE IDF VALIDATION AGAINST 2019-2025 VALIDATION SET
        print("\n" + "=" * 60)
        print("LITERATURE IDF VALIDATION ON 2019-2025 VALIDATION SET")
        print("=" * 60)
        
        lit_avg_metrics = validate_literature_idf(
            val_df, lit_data, durations, duration_hours, return_periods
        )
        
        print("\nLiterature IDF Validation Metrics on 2019-2025 (by Duration):")
        print("Average Literature Metrics Across All Durations:")
        print(lit_avg_metrics.round(4))
        
        # Save Literature validation metrics to model_performance_metrics.csv
        try:
            lit_rmse = float(lit_avg_metrics.get("RMSE", np.nan))
            lit_mae = float(lit_avg_metrics.get("MAE", np.nan))
            lit_r2 = float(lit_avg_metrics.get("R2", np.nan))
            lit_nse = float(lit_avg_metrics.get("NSE", np.nan))
            
            metrics_file_path = os.path.join(
                os.path.dirname(__file__), "..", "results", "model_performance_metrics.csv"
            )
            lit_metrics_row = {
                "Model": "Literature",
                "RMSE": lit_rmse,
                "MAE": lit_mae,
                "R2": lit_r2,
                "NSE": lit_nse,
            }
            
            if os.path.exists(metrics_file_path):
                perf_df = pd.read_csv(metrics_file_path)
                if "Literature" in perf_df["Model"].values:
                    # Update existing row
                    for col, val in lit_metrics_row.items():
                        perf_df.loc[perf_df["Model"] == "Literature", col] = val
                else:
                    perf_df = pd.concat(
                        [perf_df, pd.DataFrame([lit_metrics_row])], ignore_index=True
                    )
            else:
                perf_df = pd.DataFrame([lit_metrics_row])
            
            perf_df.to_csv(metrics_file_path, index=False)
            print(f"Literature performance metrics saved to: {metrics_file_path}")
        except Exception as e:
            print(
                f"Warning: failed to save Literature metrics to model_performance_metrics.csv: {e}"
            )
        
        # CREATE LITERATURE-ONLY IDF PLOT
        print("\nGenerating literature-only IDF curve plot...")
        plot_literature_only_idf(
            lit_data, return_periods, lit_duration_mapping, lit_avg_metrics
        )
        print(
            f"Literature IDF plot saved to: "
            f"{os.path.join(os.path.dirname(__file__), '..', 'figures', 'idf_curves_lit.png')}"
        )
        
    else:
        print(f"Literature data file not found: {lit_data_path}")
        print("Skipping literature validation and plotting.")
    
    print("\n" + "=" * 60)
    print("LOG-NORMAL ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
