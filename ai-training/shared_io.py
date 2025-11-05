import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# MinMaxScaler intentionally not imported here to avoid unused import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def shared_preprocessing():
    """
    Verbatim shared preprocessing copied from model scripts.
    Returns a dict of variables used downstream by the models so the
    model scripts can keep their existing logic unchanged.
    """
    # Read input data (path kept identical to originals)
    input_file = os.path.join(
        os.path.dirname(__file__), "..", "results", "annual_max_intensity.csv"
    )
    df = pd.read_csv(input_file)

    data_5mns = df["5mns"].values
    data_10mns = df["10mns"].values
    data_15mns = df["15mns"].values
    data_30mns = df["30mns"].values
    data_1h = df["1h"].values
    data_90min = df["90min"].values
    data_2h = df["2h"].values
    data_3h = df["3h"].values
    data_6h = df["6h"].values
    data_12h = df["12h"].values
    data_15h = df["15h"].values
    data_18h = df["18h"].values
    data_24h = df["24h"].values

    def rank_data(data):
        """Rank data using Weibull formula."""
        n = len(data)
        ranks = np.arange(1, n + 1)
        weibull_ranks = ranks / (n + 1)
        return weibull_ranks

    # Create DataFrames for each duration
    df_5mns = pd.DataFrame(
        {"duration": 5, "intensity": data_5mns, "weibull_rank": rank_data(data_5mns)}
    )
    df_10mns = pd.DataFrame(
        {"duration": 10, "intensity": data_10mns, "weibull_rank": rank_data(data_10mns)}
    )
    df_15mns = pd.DataFrame(
        {"duration": 15, "intensity": data_15mns, "weibull_rank": rank_data(data_15mns)}
    )
    df_30mns = pd.DataFrame(
        {"duration": 30, "intensity": data_30mns, "weibull_rank": rank_data(data_30mns)}
    )
    df_1h = pd.DataFrame(
        {"duration": 60, "intensity": data_1h, "weibull_rank": rank_data(data_1h)}
    )
    df_90min = pd.DataFrame(
        {"duration": 90, "intensity": data_90min, "weibull_rank": rank_data(data_90min)}
    )
    df_2h = pd.DataFrame(
        {"duration": 120, "intensity": data_2h, "weibull_rank": rank_data(data_2h)}
    )
    df_3h = pd.DataFrame(
        {"duration": 180, "intensity": data_3h, "weibull_rank": rank_data(data_3h)}
    )
    df_6h = pd.DataFrame(
        {"duration": 360, "intensity": data_6h, "weibull_rank": rank_data(data_6h)}
    )
    df_12h = pd.DataFrame(
        {"duration": 720, "intensity": data_12h, "weibull_rank": rank_data(data_12h)}
    )
    df_15h = pd.DataFrame(
        {"duration": 900, "intensity": data_15h, "weibull_rank": rank_data(data_15h)}
    )
    df_18h = pd.DataFrame(
        {"duration": 1080, "intensity": data_18h, "weibull_rank": rank_data(data_18h)}
    )
    df_24h = pd.DataFrame(
        {"duration": 1440, "intensity": data_24h, "weibull_rank": rank_data(data_24h)}
    )

    # Combine all DataFrames
    combined_df = pd.concat(
        [
            df_5mns,
            df_10mns,
            df_15mns,
            df_30mns,
            df_1h,
            df_90min,
            df_2h,
            df_3h,
            df_6h,
            df_12h,
            df_15h,
            df_18h,
            df_24h,
        ],
        ignore_index=True,
    )

    # Transform the data to make the relationship linear
    combined_df["log_duration"] = np.log(combined_df["duration"])
    combined_df["log_weibull_rank"] = np.log(combined_df["weibull_rank"])
    combined_df["log_intensity"] = np.log(combined_df["intensity"])

    # Train/validation split is handled by split_utils in the original scripts.
    # Here we only prepare commonly used constants and scalers. The model
    # scripts will call split_utils.build_train_val(...) themselves and then
    # call into shared functions below.

    duration_minutes = [5, 10, 15, 30, 60, 90, 120, 180, 360, 720, 900, 1080, 1440]
    col_names = [
        "5mns",
        "10mns",
        "15mns",
        "30mns",
        "1h",
        "90min",
        "2h",
        "3h",
        "6h",
        "12h",
        "15h",
        "18h",
        "24h",
    ]

    # Precompute durations, ranks and IDF settings used during training/validation
    return_periods = [2, 5, 10, 25, 50, 100]
    frequency_factors = {2: 0.85, 5: 1.00, 10: 1.25, 25: 1.50, 50: 1.75, 100: 2.00}

    durations = np.linspace(1, 1440, 1440)
    min_rank = combined_df["weibull_rank"].min()
    max_rank = combined_df["weibull_rank"].max()
    np.random.seed(42)
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

    out = {
        "df": df,
        "combined_df": combined_df,
        "duration_minutes": duration_minutes,
        "col_names": col_names,
        "return_periods": return_periods,
        "frequency_factors": frequency_factors,
        "durations": durations,
        "ranks": ranks,
        "standard_durations_minutes": standard_durations_minutes,
        "random_ranks": random_ranks,
    }

    return out


def build_idf_from_out_scaled(
    out_scaled,
    scaler_y,
    durations,
    frequency_factors,
    return_periods,
    standard_durations_minutes,
    csv_basename,
):
    """
    Build IDF curves and save them to CSV. This is the verbatim logic used
    across the model scripts to turn model outputs into idf curves and a CSV.
    """
    # Inverse transform and exponentiate (exact same logic as in scripts)
    base_log_intensities = scaler_y.inverse_transform(
        out_scaled.reshape(-1, 1)
    ).flatten()
    base_intensities = np.exp(base_log_intensities)

    idf_curves = {}
    for return_period in return_periods:
        idf_curves[return_period] = base_intensities * frequency_factors[return_period]

    standard_idf_curves = {}
    for return_period in return_periods:
        standard_intensities = []
        for duration in standard_durations_minutes:
            duration_idx = np.abs(durations - duration).argmin()
            standard_intensities.append(idf_curves[return_period][duration_idx])
        standard_idf_curves[return_period] = standard_intensities

    idf_df_data = {"Duration (minutes)": standard_durations_minutes}
    for rp in return_periods:
        idf_df_data[f"{rp}-year"] = standard_idf_curves[rp]

    idf_df = pd.DataFrame(idf_df_data)
    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", csv_basename)
    idf_df.to_csv(csv_path, index=False)
    print(f"IDF curves data saved to: {csv_path}")

    return {
        "idf_df": idf_df,
        "idf_curves": idf_curves,
        "standard_idf_curves": standard_idf_curves,
        "base_intensities": base_intensities,
    }


def compute_and_save_duration_metrics(
    val_df_combined,
    preds_intensity,
    duration_minutes,
    col_names,
    model_tag,
    scaler_y=None,
):
    """
    Compute per-duration validation metrics and save/update the shared
    `model_performance_metrics.csv` files.
    This implements the same logic used in the scripts, with the model name
    supplied as `model_tag` (e.g. 'ANN', 'SVM', 'TCN', 'TCAN').
    Returns the duration metrics dict and simple overall metrics tuple.
    """
    # Compute per-duration metrics
    duration_metrics = {}
    if val_df_combined is None or val_df_combined.empty:
        for col in col_names:
            duration_metrics[col] = {
                "R2": np.nan,
                "NSE": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
            }
    else:
        # Attach predictions to a copy to avoid mutating caller's frame
        vdf = val_df_combined.copy()
        vdf["pred_intensity"] = preds_intensity
        for dmin, col in zip(duration_minutes, col_names):
            mask = vdf["duration"] == dmin
            obs = vdf.loc[mask, "intensity"].values
            preds = vdf.loc[mask, "pred_intensity"].values
            if len(obs) == 0:
                duration_metrics[col] = {
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
            # NSE implementation
            obs_arr = np.array(obs)
            pred_arr = np.array(preds)
            numerator = np.sum((obs_arr - pred_arr) ** 2)
            denominator = np.sum((obs_arr - np.mean(obs_arr)) ** 2)
            nse_m = 1 - (numerator / denominator) if denominator != 0 else np.nan
            mae_m = mean_absolute_error(obs, preds)
            rmse_m = np.sqrt(mean_squared_error(obs, preds))
            duration_metrics[col] = {
                "R2": r2m,
                "NSE": nse_m,
                "MAE": mae_m,
                "RMSE": rmse_m,
            }

    # Compute overall validation metrics for model_performance_metrics.csv
    # We attempt to compute them if preds and obs are aligned globally
    overall_rmse = np.nan
    overall_mae = np.nan
    overall_r2 = np.nan
    overall_nse = np.nan
    try:
        if (
            val_df_combined is not None
            and not val_df_combined.empty
            and len(preds_intensity) == len(val_df_combined)
        ):
            y_val_intensity = val_df_combined["intensity"].values
            y_pred_intensity = preds_intensity
            overall_rmse = np.sqrt(
                mean_squared_error(y_val_intensity, y_pred_intensity)
            )
            overall_mae = mean_absolute_error(y_val_intensity, y_pred_intensity)
            overall_r2 = r2_score(y_val_intensity, y_pred_intensity)
            # nse
            numerator = np.sum((y_val_intensity - y_pred_intensity) ** 2)
            denominator = np.sum((y_val_intensity - np.mean(y_val_intensity)) ** 2)
            overall_nse = 1 - (numerator / denominator) if denominator != 0 else np.nan
    except Exception:
        pass

    metrics_row = {
        "Model": model_tag,
        "RMSE": overall_rmse,
        "MAE": overall_mae,
        "R2": overall_r2,
        "NSE": overall_nse,
    }

    metrics_file_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "model_performance_metrics.csv"
    )
    if os.path.exists(metrics_file_path):
        metrics_df = pd.read_csv(metrics_file_path)
        if model_tag in metrics_df["Model"].values:
            for col, val in metrics_row.items():
                metrics_df.loc[metrics_df["Model"] == model_tag, col] = val
        else:
            metrics_df = pd.concat(
                [metrics_df, pd.DataFrame([metrics_row])], ignore_index=True
            )
    else:
        metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(metrics_file_path, index=False)
    print(f"Model performance metrics saved to: {metrics_file_path}")

    return duration_metrics, (overall_rmse, overall_mae, overall_r2, overall_nse)


def plot_idf_comparisons(
    standard_idf_curves,
    standard_durations_minutes,
    return_periods,
    gumbel_idf,
    literature_idf,
    duration_mapping,
    literature_duration_mapping,
    model_metrics,
    gumbel_metrics,
    overall_lit_metrics,
    model_tag,
    out_prefix,
):
    """
    Create the three plotting outputs used in the scripts. Uses the same plotting
    logic as in the original files and saves images with names based on out_prefix.
    """
    # Set Times New Roman as the default font
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # model_metrics: raw validation metrics from training/evaluation (rmse, mae, r2_model, nse)
    # gumbel_metrics: metrics computed for the model-vs-gumbel comparison (overall_rmse, overall_mae, overall_r2, overall_nse)
    model_rmse, model_mae, model_r2, model_nse = (
        model_metrics if model_metrics is not None else (np.nan, np.nan, np.nan, np.nan)
    )
    gumbel_rmse, gumbel_mae, gumbel_r2, gumbel_nse = (
        gumbel_metrics
        if gumbel_metrics is not None
        else (np.nan, np.nan, np.nan, np.nan)
    )
    overall_lit_rmse, overall_lit_mae, overall_lit_r2, overall_lit_nse = (
        overall_lit_metrics
        if overall_lit_metrics is not None
        else (np.nan, np.nan, np.nan, np.nan)
    )

    # standard_durations_minutes is used directly below; no separate 'durations' variable needed

    # Comparison plot vs Gumbel
    plt.figure(figsize=(10, 6))
    colors = ["blue", "green", "red", "purple", "orange", "brown"]
    # duration_mask originally used in plotting selection; left out as unused

    def power_law(x, a, b):
        return a * x ** (-b)

    def safe_power_law_fit(x, y):
        try:
            mask = (x > 0) & (y > 0)
            if mask.sum() < 2:
                return None
            params, _ = curve_fit(power_law, x[mask], y[mask], maxfev=5000)
            x_grid = np.linspace(x.min(), x.max(), 100)
            y_fit = power_law(x_grid, *params)
            y_fit = np.maximum(y_fit, 0)
            return x_grid, y_fit
        except Exception:
            return None

    for i, rp in enumerate(return_periods):
        csv_durations = np.array(standard_durations_minutes)
        csv_intensities = np.array(standard_idf_curves[rp])
        durations_filtered = []
        intensities_filtered = []
        for j, dur_min in enumerate(csv_durations):
            if 1 <= dur_min <= 90:
                durations_filtered.append(dur_min)
                intensities_filtered.append(csv_intensities[j])
        durations_hours = np.array(durations_filtered) / 60
        intensities_filtered = np.array(intensities_filtered)
        print(
            f"📊 {model_tag} Plot {rp}-year: 5-min = {intensities_filtered[0]:.2f} mm/hr (CSV-loyal)"
        )
        sorted_indices = np.argsort(durations_hours)
        sorted_hours = durations_hours[sorted_indices]
        sorted_intensities = intensities_filtered[sorted_indices]
        fit_result = safe_power_law_fit(sorted_hours, sorted_intensities)
        if fit_result is not None:
            smooth_curve_hours, smooth_curve_intensities = fit_result
        else:
            interp_func = interp1d(
                sorted_hours,
                sorted_intensities,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            smooth_curve_hours = np.linspace(
                sorted_hours.min(), sorted_hours.max(), 100
            )
            smooth_curve_intensities = interp_func(smooth_curve_hours)
            smooth_curve_intensities = np.maximum(smooth_curve_intensities, 0)
        plt.plot(
            smooth_curve_hours,
            smooth_curve_intensities,
            "-",
            color=colors[i],
            linewidth=2,
            label=f"{model_tag} T = {rp} years",
        )
        # Gumbel
        gumbel_row = gumbel_idf[gumbel_idf["Return Period (years)"] == rp].iloc[0]
        gumbel_durations = []
        gumbel_intensities = []
        for idx, duration in enumerate(standard_durations_minutes):
            if 1 <= duration <= 90:
                gumbel_durations.append(duration / 60)
                gumbel_intensities.append(gumbel_row[duration_mapping[idx]])
        try:
            params, _ = curve_fit(
                power_law, np.array(gumbel_durations), np.array(gumbel_intensities)
            )
            smooth_durations = np.linspace(5 / 60, 1.5, 100)
            smooth_intensities = power_law(smooth_durations, *params)
            plt.plot(
                smooth_durations,
                smooth_intensities,
                "--",
                color=colors[i],
                linewidth=1.5,
                label=f"Gumbel T = {rp} years",
            )
        except RuntimeError:
            plt.plot(
                np.array(gumbel_durations),
                gumbel_intensities,
                "--",
                color=colors[i],
                linewidth=1.5,
                label=f"Gumbel T = {rp} years",
            )

    tick_positions = [15 / 60, 30 / 60, 45 / 60, 1, 1.25, 1.5]
    tick_labels = ["15min", "30min", "45min", "1hr", "1.25hr", "1.5hr"]
    plt.xticks(tick_positions, tick_labels)
    plt.xlabel("Duration (hours)", fontsize=12)
    plt.ylabel("Intensity (mm/hr)", fontsize=12)
    plt.title(f"IDF Curves Comparison: {model_tag} vs Gumbel", fontsize=14)
    plt.grid(True)
    # Use gumbel_metrics for the Gumbel-comparison annotation box
    plt.text(
        0.02,
        0.98,
        f"RMSE: {gumbel_rmse:.4f}\nMAE: {gumbel_mae:.4f}\nR²: {gumbel_r2:.4f}\nNSE: {gumbel_nse:.4f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "figures", f"idf_comparison_{out_prefix}.png"
    )
    plt.savefig(out_path, dpi=300)
    print(f"Comparison plot saved to: {out_path}")

    # Literature plot
    plt.figure(figsize=(10, 6))
    for i, rp in enumerate(return_periods):
        csv_durations = np.array(standard_durations_minutes)
        csv_intensities = np.array(standard_idf_curves[rp])
        durations_filtered = []
        intensities_filtered = []
        for j, dur_min in enumerate(csv_durations):
            if 5 <= dur_min <= 90:
                durations_filtered.append(dur_min)
                intensities_filtered.append(csv_intensities[j])
        durations_hours = np.array(durations_filtered) / 60
        intensities_filtered = np.array(intensities_filtered)
        print(
            f"📊 {model_tag} Literature Plot {rp}-year: 5-min = {intensities_filtered[0]:.2f} mm/hr (CSV-loyal)"
        )
        sorted_indices = np.argsort(durations_hours)
        sorted_hours = durations_hours[sorted_indices]
        sorted_intensities = intensities_filtered[sorted_indices]
        fit_result = safe_power_law_fit(sorted_hours, sorted_intensities)
        if fit_result is not None:
            smooth_curve_hours, smooth_curve_intensities = fit_result
        else:
            interp_func = interp1d(
                sorted_hours,
                sorted_intensities,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            smooth_curve_hours = np.linspace(
                sorted_hours.min(), sorted_hours.max(), 100
            )
            smooth_curve_intensities = interp_func(smooth_curve_hours)
            smooth_curve_intensities = np.maximum(smooth_curve_intensities, 0)
        plt.plot(
            smooth_curve_hours,
            smooth_curve_intensities,
            "-",
            color=colors[i],
            linewidth=2,
            label=f"{model_tag} T = {rp} years",
        )
        lit_row = literature_idf[literature_idf["Return Period (years)"] == rp].iloc[0]
        lit_durations = []
        lit_intensities = []
        for duration in standard_durations_minutes:
            if 5 <= duration <= 120:
                lit_col = literature_duration_mapping[duration]
                lit_value = lit_row[lit_col]
                if pd.notna(lit_value) and lit_value != "":
                    lit_durations.append(duration / 60)
                    lit_intensities.append(float(lit_value))
        if len(lit_durations) >= 3:
            try:
                params, _ = curve_fit(
                    power_law, np.array(lit_durations), np.array(lit_intensities)
                )
                smooth_durations = np.linspace(5 / 60, 1.5, 100)
                smooth_intensities = power_law(smooth_durations, *params)
                plt.plot(
                    smooth_durations,
                    smooth_intensities,
                    "--",
                    color=colors[i],
                    linewidth=1.5,
                    label=f"Literature T = {rp} years",
                )
            except RuntimeError:
                plt.scatter(
                    lit_durations,
                    lit_intensities,
                    color=colors[i],
                    marker="o",
                    s=30,
                    label=f"Literature T = {rp} years",
                )
        elif len(lit_durations) > 0:
            plt.scatter(
                lit_durations,
                lit_intensities,
                color=colors[i],
                marker="o",
                s=30,
                label=f"Literature T = {rp} years",
            )

    plt.xticks(tick_positions, tick_labels)
    plt.xlabel("Duration (hours)", fontsize=12)
    plt.ylabel("Intensity (mm/hr)", fontsize=12)
    plt.title(f"IDF Curves Comparison: {model_tag} vs Literature", fontsize=14)
    plt.grid(True)
    if not np.isnan(overall_lit_rmse):
        plt.text(
            0.02,
            0.98,
            f"RMSE: {overall_lit_rmse:.4f}\nMAE: {overall_lit_mae:.4f}\nR²: {overall_lit_r2:.4f}\nNSE: {overall_lit_nse:.4f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_path2 = os.path.join(
        os.path.dirname(__file__),
        "..",
        "figures",
        f"idf_comparison_{out_prefix}_literature.png",
    )
    plt.savefig(out_path2, dpi=300)
    print(f"Literature comparison plot saved to: {out_path2}")

    # Original IDF curve plot using CSV data for consistency
    plt.figure(figsize=(10, 6))
    for return_period in return_periods:
        csv_durations = np.array(standard_durations_minutes)
        csv_intensities = np.array(standard_idf_curves[return_period])
        durations_filtered = []
        intensities_filtered = []
        for j, dur_min in enumerate(csv_durations):
            if 1 <= dur_min <= 90:
                durations_filtered.append(dur_min)
                intensities_filtered.append(csv_intensities[j])
        durations_hours = np.array(durations_filtered) / 60
        intensities_filtered = np.array(intensities_filtered)
        print(
            f"📊 {model_tag} IDF Plot {return_period}-year: 5-min = {intensities_filtered[0]:.2f} mm/hr (CSV-loyal)"
        )
        sorted_indices = np.argsort(durations_hours)
        sorted_hours = durations_hours[sorted_indices]
        sorted_intensities = intensities_filtered[sorted_indices]
        fit_result = safe_power_law_fit(sorted_hours, sorted_intensities)
        if fit_result is not None:
            smooth_curve_hours, smooth_curve_intensities = fit_result
        else:
            interp_func = interp1d(
                sorted_hours,
                sorted_intensities,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            smooth_curve_hours = np.linspace(
                sorted_hours.min(), sorted_hours.max(), 100
            )
            smooth_curve_intensities = interp_func(smooth_curve_hours)
            smooth_curve_intensities = np.maximum(smooth_curve_intensities, 0)
        plt.plot(
            smooth_curve_hours,
            smooth_curve_intensities,
            linewidth=2,
            label=f"{return_period}-year return period",
        )

    plt.xticks(tick_positions, tick_labels)
    plt.xlabel("Duration (hours)")
    plt.ylabel("Intensity (mm/h)")
    plt.title(f"Intensity-Duration-Frequency (IDF) Curves using {model_tag}")
    plt.grid(True)
    # Use model_metrics (raw validation metrics) for the IDF-curves annotation box
    plt.text(
        0.02,
        0.98,
        f"MAE = {model_mae:.4f}\nRMSE = {model_rmse:.4f}\nR² = {model_r2:.4f}\nNSE = {model_nse:.4f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.legend()
    plt.tight_layout()
    out_path3 = os.path.join(
        os.path.dirname(__file__), "..", "figures", f"idf_curves_{out_prefix}.png"
    )
    plt.savefig(out_path3, dpi=300)
    plt.show()
    print(f"IDF curves plot saved to: {out_path3}")
