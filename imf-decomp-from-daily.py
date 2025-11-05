import os
import pandas as pd
import numpy as np

# Read the daily CSV file
file_path = os.path.join(
    os.path.dirname(__file__), "data", "gpm-bey-daily.csv"
)
df = pd.read_csv(file_path, parse_dates=["date"])


# Function to apply IMF decomposition for intensities
def apply_imf_decomp_intensity(intensity_T, t, T, n=(1/3)):
    """
    Apply IMF decomposition for intensities using Indian Meteorological Formula.

    Parameters:
    intensity_T (float): Rainfall intensity for duration T in mm/hr
    t (float): Target duration in minutes (or hours for longer durations)
    T (float): Original duration in minutes (or hours for longer durations) 
    n (float): Empirical parameter for Bell's ratio (default 1/3)

    Returns:
    float: Rainfall intensity for duration t in mm/hr
    """
    if intensity_T == 0:
        return 0.0

    # For intensities, the relationship is I_t / I_T = (T/t)^(1-n)
    # Ensure all calculations use float64 precision
    ratio = np.float64(T / t) ** np.float64(1 - n)
    intensity_t = np.float64(intensity_T) * ratio

    return float(intensity_t)


def create_disaggregated_dataset(dates, daily_precip, target_duration_min, intervals_per_day):
    """
    Create disaggregated dataset for a given target duration.
    
    Parameters:
    dates: Array of daily dates
    daily_precip: Array of daily precipitation values (mm/day)
    target_duration_min: Target duration in minutes
    intervals_per_day: Number of intervals per day for this duration
    
    Returns:
    tuple: (dates_array, values_array) for the disaggregated data
    """
    n_rows = len(dates)
    n_intervals = n_rows * intervals_per_day
    
    # Pre-allocate arrays
    new_dates = np.zeros(n_intervals, dtype="datetime64[ns]")
    new_values = np.zeros(n_intervals, dtype=np.float64)
    
    # Convert daily precipitation to daily intensity (mm/hr)
    daily_intensities = daily_precip / 24.0
    
    # Calculate target intensity using IMF decomposition
    target_intensities = np.vectorize(
        lambda x: apply_imf_decomp_intensity(x, target_duration_min, 24*60), 
        otypes=[np.float64]
    )(daily_intensities)
    
    # Fill arrays for each interval within the day
    for i in range(intervals_per_day):
        idx = np.arange(i, n_intervals, intervals_per_day)
        new_dates[idx] = dates + np.timedelta64(i * target_duration_min, "m")
        new_values[idx] = target_intensities
    
    return new_dates, new_values


def create_aggregated_dataset(dates, daily_precip, target_duration_hours):
    """
    Create aggregated dataset for durations longer than 24 hours.
    
    Parameters:
    dates: Array of daily dates
    daily_precip: Array of daily precipitation values (mm/day)
    target_duration_hours: Target duration in hours
    
    Returns:
    tuple: (dates_array, values_array) for the aggregated data
    """
    # Convert daily precipitation to daily intensity (mm/hr)
    daily_intensities = daily_precip / 24.0
    
    # Calculate target intensity using IMF decomposition
    target_intensities = np.vectorize(
        lambda x: apply_imf_decomp_intensity(x, target_duration_hours, 24), 
        otypes=[np.float64]
    )(daily_intensities)
    
    return dates, target_intensities


# Extract dates and precipitation values
dates = df["date"].values
daily_precip_values = df["value"].values.astype(np.float64)

print("Starting IMF decomposition from daily data...")
print(f"Processing {len(df)} days of data")

# Define all target durations and their properties
# Format: (duration_name, duration_minutes, intervals_per_day, filename)
disaggregation_configs = [
    ("5min", 5, 288, "gpm-bey-5mns.csv"),
    ("10min", 10, 144, "gpm-bey-10mns.csv"),
    ("15min", 15, 96, "gpm-bey-15mns.csv"),
    ("30min", 30, 48, "gpm-bey-30mns.csv"),
    ("1hr", 60, 24, "gpm-bey-1hr.csv"),
    ("90min", 90, 16, "gpm-bey-90min.csv"),
    ("2hr", 120, 12, "gpm-bey-2hr.csv"),
    ("3hr", 180, 8, "gpm-bey-3hr.csv"),
    ("6hr", 360, 4, "gpm-bey-6hr.csv"),
    ("12hr", 720, 2, "gpm-bey-12hr.csv"),
]

# Define aggregation configurations (durations longer than 24 hours)
# Format: (duration_name, duration_hours, filename)
aggregation_configs = [
    ("15hr", 15, "gpm-bey-15hr.csv"),
    ("18hr", 18, "gpm-bey-18hr.csv"),
]

# Process disaggregated datasets (durations ≤ 12 hours)
print("\nProcessing disaggregated datasets...")
for duration_name, duration_min, intervals_per_day, filename in disaggregation_configs:
    print(f"Creating {duration_name} dataset...")
    
    new_dates, new_values = create_disaggregated_dataset(
        dates, daily_precip_values, duration_min, intervals_per_day
    )
    
    # Create DataFrame and save
    df_new = pd.DataFrame({"date": new_dates, "value": new_values})
    output_path = os.path.join("./data", filename)
    df_new.to_csv(output_path, index=False)
    
    print(f"  - Shape: {df_new.shape}")
    print(f"  - Max date: {df_new['date'].max()}")
    print(f"  - Saved to: {filename}")

# Process aggregated datasets (durations between 12-24 hours)
print("\nProcessing aggregated datasets...")
for duration_name, duration_hours, filename in aggregation_configs:
    print(f"Creating {duration_name} dataset...")
    
    new_dates, new_values = create_aggregated_dataset(
        dates, daily_precip_values, duration_hours
    )
    
    # Create DataFrame and save
    df_new = pd.DataFrame({"date": new_dates, "value": new_values})
    output_path = os.path.join("./data", filename)
    df_new.to_csv(output_path, index=False)
    
    print(f"  - Shape: {df_new.shape}")
    print(f"  - Max date: {df_new['date'].max()}")
    print(f"  - Saved to: {filename}")

# Also create the daily dataset (as intensity rather than precipitation)
print("\nCreating daily intensity dataset...")
daily_intensities = daily_precip_values / 24.0
df_daily_intensity = pd.DataFrame({"date": dates, "value": daily_intensities})
df_daily_intensity.to_csv("./data/gpm-bey-daily-intensity.csv", index=False)
print(f"  - Shape: {df_daily_intensity.shape}")
print("  - Saved to: gpm-bey-daily-intensity.csv")

print("\n" + "="*60)
print("IMF DECOMPOSITION COMPLETE")
print("="*60)
print("All datasets generated from 24-hour (daily) data using Indian Meteorological Formula")
print(f"Original daily data: {df.shape[0]} days")
print("\nGenerated datasets:")

print("\nDisaggregated datasets (sub-daily):")
for duration_name, duration_min, intervals_per_day, filename in disaggregation_configs:
    total_intervals = len(df) * intervals_per_day
    print(f"  - {duration_name:>6}: {total_intervals:>8} intervals ({filename})")

print("\nAggregated datasets (sub-daily to daily):")
for duration_name, duration_hours, filename in aggregation_configs:
    print(f"  - {duration_name:>6}: {len(df):>8} intervals ({filename})")

print(f"\nDaily intensity: {len(df):>8} intervals (gpm-bey-daily-intensity.csv)")

print("\nAll files saved to ./data/ directory")
