import os
import pandas as pd
import numpy as np

# Read the daily CSV file
file_path = os.path.join(
    os.path.dirname(__file__), "data", "gpm-bey-daily.csv"
)
df = pd.read_csv(file_path, parse_dates=["date"])
# No conversion to intensity here - work with precipitation values directly


# Function to apply IMF decomposition for intensities
def apply_imf_decomp_intensity(intensity_T, t, T, n=(1/3)):
    """
    Apply IMF decomposition for intensities.

    Parameters:
    intensity_T (float): Rainfall intensity for duration T in mm/hr
    t (float): Target duration in hours
    T (float): Original duration in hours
    n (float): Empirical parameter for Bell's ratio

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


# Create DataFrames for each target duration
df_12hr = pd.DataFrame(columns=["date", "value"])
df_15hr = pd.DataFrame(columns=["date", "value"])  # New 15hr DataFrame
df_18hr = pd.DataFrame(columns=["date", "value"])  # New 18hr DataFrame

# For each 24-hour interval, create higher resolution data
# Calculate size of output arrays
n_rows = len(df)
n_12hr = n_rows * 2
n_15hr = n_rows  # One 15hr interval per day
n_18hr = n_rows  # One 18hr interval per day

# Pre-allocate arrays with explicit float64 dtype for precision
dates_12hr = np.zeros(n_12hr, dtype="datetime64[ns]")
values_12hr = np.zeros(n_12hr, dtype=np.float64)
dates_15hr = np.zeros(n_15hr, dtype="datetime64[ns]")
values_15hr = np.zeros(n_15hr, dtype=np.float64)
dates_18hr = np.zeros(n_18hr, dtype="datetime64[ns]")
values_18hr = np.zeros(n_18hr, dtype=np.float64)

# Extract dates and precipitation values as arrays with explicit dtypes
dates = df["date"].values
daily_precip_values = df["value"].values.astype(np.float64)

# Convert daily precipitation (mm/day) to daily intensity (mm/hr)
daily_intensities = daily_precip_values / 24.0  # mm/day ÷ 24 hours = mm/hr

# Calculate all intensities at once using IMF decomposition with explicit float64
intensity_12hr = np.vectorize(lambda x: apply_imf_decomp_intensity(x, 12, 24), otypes=[np.float64])(
    daily_intensities
)
intensity_15hr = np.vectorize(lambda x: apply_imf_decomp_intensity(x, 15, 24), otypes=[np.float64])(
    daily_intensities
)
intensity_18hr = np.vectorize(lambda x: apply_imf_decomp_intensity(x, 18, 24), otypes=[np.float64])(
    daily_intensities
)

# Fill arrays for 12 hour intervals
for i in range(2):
    idx = np.arange(i, n_12hr, 2)
    dates_12hr[idx] = dates + np.timedelta64(i * 720, "m")
    values_12hr[idx] = intensity_12hr

# Fill arrays for 15 hour intervals
for i in range(1):  # Just one interval per day
    dates_15hr = dates + np.timedelta64(i * 15 * 60, "m")
    values_15hr = intensity_15hr

# Fill arrays for 18 hour intervals
for i in range(1):  # Just one interval per day
    dates_18hr = dates + np.timedelta64(i * 18 * 60, "m")
    values_18hr = intensity_18hr

# Create DataFrames
df_12hr = pd.DataFrame({"date": dates_12hr, "value": values_12hr})
df_15hr = pd.DataFrame({"date": dates_15hr, "value": values_15hr})
df_18hr = pd.DataFrame({"date": dates_18hr, "value": values_18hr})

# Print the max date for each dataframe
print(f"Original daily data max date: {df['date'].max()}")
print(f"12hr data max date: {df_12hr['date'].max()}")
print(f"15hr data max date: {df_15hr['date'].max()}")
print(f"18hr data max date: {df_18hr['date'].max()}")

# Print some statistics
print("\nStatistics:")
print(f"Original daily data shape: {df.shape}")
print(f"12hr data shape: {df_12hr.shape}")
print(f"15hr data shape: {df_15hr.shape}")
print(f"18hr data shape: {df_18hr.shape}")

# Save the results
df_12hr.to_csv("./data/gpm-bey-12hr.csv", index=False)
df_15hr.to_csv("./data/gpm-bey-15hr.csv", index=False)
df_18hr.to_csv("./data/gpm-bey-18hr.csv", index=False)

print("\nDisaggregation complete. Files saved:")
print("- gpm-bey-12hr.csv")
print("- gpm-bey-15hr.csv")
print("- gpm-bey-18hr.csv")
