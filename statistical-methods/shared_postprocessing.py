import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error


def power_law(x, a, b):
    """Power law function for fitting IDF curves: I = a * D^(-b)"""
    return a * x ** (-b)


def r2_score(observed, simulated):
    """Calculate squared Pearson correlation coefficient (R²) between observed and simulated values."""
    observed = np.array(observed)
    simulated = np.array(simulated)
    if len(observed) != len(simulated) or len(observed) == 0:
        return np.nan
    correlation_matrix = np.corrcoef(observed, simulated)
    if correlation_matrix.shape != (2, 2):
        return np.nan
    r = correlation_matrix[0, 1]
    return r**2


def nash_sutcliffe_efficiency(observed, simulated):
    """
    Compute Nash-Sutcliffe Efficiency (NSE).

    Parameters:
        observed (array-like): Array of observed values.
        simulated (array-like): Array of simulated values.

    Returns:
        float: Nash-Sutcliffe Efficiency coefficient.
    """
    observed = np.array(observed)
    simulated = np.array(simulated)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else np.nan


def save_performance_metrics(
    model_name, rmse, mae, r2, nse, filename="model_performance_metrics.csv"
):
    """Save or update performance metrics for a model."""
    metrics_file_path = os.path.join(
        os.path.dirname(__file__), "..", "results", filename
    )
    metrics_row = {
        "Model": model_name,
        "RMSE": float(rmse) if not pd.isna(rmse) else np.nan,
        "MAE": float(mae) if not pd.isna(mae) else np.nan,
        "R2": float(r2) if not pd.isna(r2) else np.nan,
        "NSE": float(nse) if not pd.isna(nse) else np.nan,
    }

    if os.path.exists(metrics_file_path):
        perf_df = pd.read_csv(metrics_file_path)
        # Handle legacy 'Gumbel' model name migration to 'Statistical'
        if (
            model_name == "Statistical"
            and "Gumbel" in perf_df.get("Model", pd.Series(dtype=str)).values
        ):
            if "Statistical" in perf_df["Model"].values:
                perf_df = perf_df[perf_df["Model"] != "Gumbel"].copy()
            else:
                perf_df.loc[perf_df["Model"] == "Gumbel", "Model"] = "Statistical"

        if model_name in perf_df["Model"].values:
            for col, val in metrics_row.items():
                perf_df.loc[perf_df["Model"] == model_name, col] = val
        else:
            perf_df = pd.concat(
                [perf_df, pd.DataFrame([metrics_row])], ignore_index=True
            )
    else:
        perf_df = pd.DataFrame([metrics_row])

    perf_df.to_csv(metrics_file_path, index=False)
    print(f"{model_name} performance metrics saved to: {metrics_file_path}")


def _format_idf_plot(title, metrics=None):
    """Apply common formatting to IDF plots."""
    tick_positions = [15 / 60, 30 / 60, 45 / 60, 1, 1.25, 1.5]
    tick_labels = ["15min", "30min", "45min", "1hr", "1.25hr", "1.5hr"]
    plt.xticks(tick_positions, tick_labels)
    plt.xlabel("Duration (hours)")
    plt.ylabel("Rainfall Intensity (mm/hr)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Add metrics overlay if provided
    if metrics is not None:
        metrics_text = f"RMSE: {metrics['RMSE']:.4f}\nMAE: {metrics['MAE']:.4f}\nR²: {metrics['R2']:.4f}\nNSE: {metrics['NSE']:.4f}"
        plt.text(
            0.02,
            0.98,
            metrics_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
        )
    plt.tight_layout()


def create_idf_plot(
    intensities_data,
    return_periods,
    duration_hours,
    colors,
    title,
    filename,
    metrics=None,
    show_power_law_params=False,
):
    """Create IDF curves plot with common formatting."""
    # Set Times New Roman as the default font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(10, 6))

    # Filter durations from 5 minutes to 1.5 hours
    duration_range = slice(0, 6)  # 5min to 1.5hr
    filtered_duration_hours = duration_hours[duration_range]
    filtered_intensities = (
        intensities_data[:, duration_range]
        if intensities_data.ndim > 1
        else intensities_data
    )

    # Create smooth curves for each return period
    x_fine = np.linspace(
        min(filtered_duration_hours), max(filtered_duration_hours), 100
    )

    for i, rp in enumerate(return_periods):
        if intensities_data.ndim > 1:
            current_intensities = filtered_intensities[i]
        else:
            current_intensities = filtered_intensities

        # Fit power law in log-log space
        log_durations = np.log(filtered_duration_hours)
        log_intensities = np.log(current_intensities)

        slope, intercept = np.polyfit(log_durations, log_intensities, 1)

        # Generate smooth curve
        log_x_fine = np.log(x_fine)
        log_y_fine = intercept + slope * log_x_fine
        y_fine = np.exp(log_y_fine)

        plt.plot(x_fine, y_fine, label=f"{rp}-year", color=colors[i], linewidth=2)

        if show_power_law_params:
            c = np.exp(intercept)
            n = -slope
            print(f"Return Period {rp} years: I = {c:.2f}/(D^{n:.2f})")

    # Common formatting
    _format_idf_plot(title, metrics)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.show()


def _extract_literature_data(lit_row, lit_duration_mapping, duration_minutes_list):
    """Extract literature data for given duration list."""
    lit_durations, lit_intensities = [], []

    for duration_min in duration_minutes_list:
        if duration_min == 90:  # Skip 90min as not in literature
            continue
        if duration_min in lit_duration_mapping:
            lit_col = lit_duration_mapping[duration_min]
            lit_value = lit_row[lit_col]

            if not (pd.isna(lit_value) or lit_value == "" or lit_value == 0):
                lit_durations.append(duration_min / 60)  # Convert to hours
                lit_intensities.append(float(lit_value))

    return lit_durations, lit_intensities


def plot_model_vs_literature_comparison(
    intensities_data,
    lit_data,
    return_periods,
    duration_hours,
    lit_duration_mapping,
    model_name="Model",
    filename="comparison.png",
    metrics=None,
):
    """Create comparison plot between model predictions and literature data."""
    # Set Times New Roman as the default font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(10, 6))
    colors = ["blue", "green", "red", "purple", "orange", "brown"]

    # Filter durations
    duration_mask = (np.array(duration_hours) >= 5 / 60) & (
        np.array(duration_hours) <= 1.5
    )
    filtered_duration_hours = np.array(duration_hours)[duration_mask]

    for i, rp in enumerate(return_periods):
        # Plot model data
        filtered_intensities = intensities_data[i][duration_mask]

        try:
            model_params, _ = curve_fit(
                power_law, filtered_duration_hours, filtered_intensities
            )
            smooth_durations = np.linspace(5 / 60, 1.5, 100)
            smooth_model_intensities = power_law(smooth_durations, *model_params)
            plt.plot(
                smooth_durations,
                smooth_model_intensities,
                "-",
                color=colors[i],
                linewidth=2,
                label=f"{model_name} T = {rp} years",
            )
        except RuntimeError:
            plt.plot(
                filtered_duration_hours,
                filtered_intensities,
                "-",
                color=colors[i],
                linewidth=2,
                label=f"{model_name} T = {rp} years",
            )

        # Plot literature data
        lit_row = lit_data[lit_data["Return Period (years)"] == rp].iloc[0]
        lit_durations, lit_intensities = _extract_literature_data(
            lit_row, lit_duration_mapping, [5, 10, 15, 30, 60]
        )

        if len(lit_durations) >= 2:
            try:
                lit_params, _ = curve_fit(
                    power_law, np.array(lit_durations), np.array(lit_intensities)
                )
                smooth_durations = np.linspace(5 / 60, 1.5, 100)
                smooth_lit_intensities = power_law(smooth_durations, *lit_params)
                plt.plot(
                    smooth_durations,
                    smooth_lit_intensities,
                    "--",
                    color=colors[i],
                    linewidth=1.5,
                    label=f"Literature T = {rp} years",
                )
            except RuntimeError:
                plt.plot(
                    np.array(lit_durations),
                    lit_intensities,
                    "--",
                    color=colors[i],
                    linewidth=1.5,
                    label=f"Literature T = {rp} years",
                )

    _format_idf_plot(f"IDF Curves Comparison: {model_name} vs Literature", metrics)
    plt.legend(loc="upper right", fontsize=9)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.show()


def plot_literature_only_idf(
    lit_data,
    return_periods,
    lit_duration_mapping,
    metrics=None,
    filename="idf_curves_lit.png",
):
    """Create literature-only IDF plot."""
    # Set Times New Roman as the default font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(10, 6))
    colors = ["blue", "green", "red", "purple", "orange", "brown"]

    for i, rp in enumerate(return_periods):
        lit_row = lit_data[lit_data["Return Period (years)"] == rp].iloc[0]
        lit_durations, lit_intensities = _extract_literature_data(
            lit_row, lit_duration_mapping, [5, 10, 15, 30, 60]
        )

        if len(lit_durations) >= 2:
            try:
                params, _ = curve_fit(
                    power_law, np.array(lit_durations), np.array(lit_intensities)
                )
                smooth_durations = np.linspace(5 / 60, 1.5, 100)
                smooth_intensities = power_law(smooth_durations, *params)
                plt.plot(
                    smooth_durations,
                    smooth_intensities,
                    "-",
                    color=colors[i],
                    linewidth=2,
                    label=f"{rp}-year",
                )

                # Print literature power law parameters
                a, b = params
                print(
                    f"Literature Return Period {rp} years: I = {a:.2f} × D^(-{b:.2f})"
                )
            except RuntimeError:
                plt.plot(
                    np.array(lit_durations),
                    lit_intensities,
                    "o-",
                    color=colors[i],
                    linewidth=2,
                    label=f"{rp}-year",
                )

    _format_idf_plot("Literature IDF Curves", metrics)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.show()


def compare_with_literature(idf_data, lit_data, return_periods, model_name="Model"):
    """Compare model results with literature data and calculate metrics."""
    # Duration mapping for comparison
    available_durations_minutes = [5, 10, 15, 30, 60, 120, 180, 360, 720, 1440]

    lit_duration_mapping = {
        5: "5 mins",
        10: "10 mins",
        15: "15 mins",
        30: "30 mins",
        60: "60 mins",
        120: "120 mins",
        180: "180 mins",
        360: "360 mins",
        720: "720 mins",
        1440: "1440 mins",
    }

    our_duration_mapping = {
        5: "5 mins",
        10: "10 mins",
        15: "15 mins",
        30: "30 mins",
        60: "60 mins",
        120: "120 mins",
        180: "180 mins",
        360: "360 mins",
        720: "720 mins",
        1440: "1440 mins",
    }

    # Calculate metrics for each return period
    rmse_values, mae_values, r2_values, nse_values = [], [], [], []

    for rp in return_periods:
        lit_row = lit_data[lit_data["Return Period (years)"] == rp].iloc[0]
        our_row = idf_data.loc[rp]

        y_true, y_pred = [], []

        for duration_min in available_durations_minutes:
            lit_col = lit_duration_mapping[duration_min]
            lit_value = lit_row[lit_col]

            if pd.isna(lit_value) or lit_value == "" or lit_value == 0:
                continue

            our_col = our_duration_mapping[duration_min]
            our_value = our_row[our_col]

            y_true.append(float(lit_value))
            y_pred.append(float(our_value))

        if len(y_true) > 0:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            nse = nash_sutcliffe_efficiency(y_true, y_pred)

            rmse_values.extend([rmse])
            mae_values.extend([mae])
            r2_values.extend([r2])
            nse_values.extend([nse])
        else:
            rmse_values.extend([np.nan])
            mae_values.extend([np.nan])
            r2_values.extend([np.nan])
            nse_values.extend([np.nan])

    # Calculate overall metrics
    valid_indices = ~np.isnan(rmse_values)
    overall_rmse = (
        np.mean(np.array(rmse_values)[valid_indices])
        if np.any(valid_indices)
        else np.nan
    )
    overall_mae = (
        np.mean(np.array(mae_values)[valid_indices])
        if np.any(valid_indices)
        else np.nan
    )
    overall_r2 = (
        np.mean(np.array(r2_values)[valid_indices]) if np.any(valid_indices) else np.nan
    )
    overall_nse = (
        np.mean(np.array(nse_values)[valid_indices])
        if np.any(valid_indices)
        else np.nan
    )

    print(f"\n{model_name} vs Literature - Overall Performance:")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE: {overall_mae:.4f}")
    print(f"R2: {overall_r2:.4f}")
    print(f"NSE: {overall_nse:.4f}")

    return {
        "RMSE": overall_rmse,
        "MAE": overall_mae,
        "R2": overall_r2,
        "NSE": overall_nse,
    }


def get_literature_duration_mapping():
    """Get the standard literature duration mapping."""
    return {
        5: "5 mins",
        10: "10 mins",
        15: "15 mins",
        30: "30 mins",
        60: "60 mins",
        120: "120 mins",
        180: "180 mins",
        360: "360 mins",
        720: "720 mins",
        1440: "1440 mins",
    }
