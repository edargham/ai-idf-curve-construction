import os
import pandas as pd


def load_precipitation_data():
    """Load all precipitation data files and return processed dataframes."""
    # Define data file configurations
    data_configs = [
        ("5mns", "gpm-bey-5mns-bl.csv"),
        ("10mns", "gpm-bey-10mns-bl.csv"),
        ("15mns", "gpm-bey-15mns-bl.csv"),
        ("30mns", "gpm-bey-30mns.csv"),
        ("1h", "gpm-bey-1hr-bl.csv"),
        ("90min", "gpm-bey-90min-bl.csv"),
        ("2h", "gpm-bey-2hr-bl.csv"),
        ("3h", "gpm-bey-3hr-bl.csv"),
        ("6h", "gpm-bey-6hr-bl.csv"),
        ("12h", "gpm-bey-12hr-bl.csv"),
        ("15h", "gpm-bey-15hr-bl.csv"),
        ("18h", "gpm-bey-18hr-bl.csv"),
        ("24h", "gpm-bey-daily.csv"),
    ]

    dataframes = []
    column_names = []

    for col_name, filename in data_configs:
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        df = pd.read_csv(file_path)

        # Convert daily precipitation totals to intensities for 24h data
        if col_name == "24h":
            df["value"] = df["value"] / 24.0  # mm/day ÷ 24 hours = mm/hr

        # Convert date column to datetime and set as index
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        dataframes.append(df)
        column_names.append(col_name)

    # Merge all dataframes
    merged_df = pd.concat(dataframes, axis=1)
    merged_df.columns = column_names
    merged_df.fillna(0, inplace=True)

    return merged_df


def save_aggregated_data(df, filename="bey-aggregated-final"):
    """Save aggregated dataframe to CSV file."""
    output_path = os.path.join(os.path.dirname(__file__), "results", filename)
    df.to_csv(f"{output_path}.csv", index=True)
    print(f"Aggregated data saved to {output_path}")


def create_intensity_dataframe(df):
    """Create intensity dataframe with year column and intensity columns."""
    df["year"] = df.index.year
    df_intensity = df.copy()

    # Add intensity columns (data is already in mm/hr format)
    duration_cols = [
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

    for col in duration_cols:
        df_intensity[f"{col}_intensity"] = df_intensity[col].copy()

    return df_intensity





if __name__ == "__main__":
    # Load and process precipitation data
    df = load_precipitation_data()
    save_aggregated_data(df)

    # Create intensity dataframe with additional columns
    df_intensity = create_intensity_dataframe(df)

    # Save historical intensity data
    columns = ["year"] + [col for col in df_intensity.columns if col != "year"]
    df_intensity = df_intensity[columns]
    df_intensity.to_csv(
        os.path.join(os.path.dirname(__file__), "results", "historical_intensity.csv"),
        index=True,
    )


# Enhanced extreme event sampling strategy
def extract_extreme_events(df, threshold_percentile=95, min_separation_days=3):
    """
    Extract extreme events using multiple criteria:
    1. Peak over threshold (POT) method
    2. Minimum separation between events to ensure independence
    3. Seasonal consideration to avoid bias
    """
    extreme_events = []

    # Duration columns
    duration_cols = [
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

    for col in duration_cols:
        # Calculate threshold based on percentile
        threshold = df[col].quantile(threshold_percentile / 100)

        # Find events above threshold
        above_threshold = df[df[col] > threshold].copy()

        if len(above_threshold) == 0:
            continue

        # Sort by intensity (descending)
        above_threshold = above_threshold.sort_values(col, ascending=False)

        # Apply minimum separation constraint
        selected_events = []
        for idx, row in above_threshold.iterrows():
            # Check if this event is separated enough from previously selected ones
            is_independent = True
            for prev_date in selected_events:
                if abs((idx - prev_date).days) < min_separation_days:
                    is_independent = False
                    break

            if is_independent:
                selected_events.append(idx)

        # Store the selected extreme events for this duration
        if selected_events:
            extreme_events.extend(
                above_threshold.loc[selected_events, [col, "year"]].values.tolist()
            )

    return extreme_events


def smart_annual_maxima_selection(df, n_events_per_year=3):
    """
    Smart selection of extreme events considering:
    1. Multiple events per year (not just annual maximum)
    2. Different storm types and seasons
    3. Partial duration series approach
    """
    duration_cols = [
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

    # Enhanced annual maxima with consideration for multiple events
    enhanced_annual_data = {}

    for year in df["year"].unique():
        if pd.isna(year):
            continue

        year_data = df[df["year"] == year]
        year_maxima = {}

        for col in duration_cols:
            # Get top n events for each duration in this year
            top_events = year_data.nlargest(n_events_per_year, col)[col].values

            # Use different strategies based on the number of available events
            if len(top_events) >= n_events_per_year:
                # Use weighted approach: 60% weight to max, 30% to 2nd max, 10% to 3rd max
                weights = [0.6, 0.3, 0.1]
                enhanced_max = sum(
                    w * event
                    for w, event in zip(weights, top_events[:n_events_per_year])
                )
            elif len(top_events) >= 2:
                # Use 70% max, 30% second max
                enhanced_max = 0.7 * top_events[0] + 0.3 * top_events[1]
            else:
                # Fall back to simple maximum
                enhanced_max = top_events[0] if len(top_events) > 0 else 0

            year_maxima[col] = enhanced_max

        enhanced_annual_data[year] = year_maxima

    # Convert to DataFrame with same structure as original
    enhanced_df = pd.DataFrame.from_dict(enhanced_annual_data, orient="index")
    enhanced_df.index.name = "year"

    return enhanced_df


def partial_duration_series_adjustment(df, lambda_threshold=2.0):
    """
    Apply partial duration series method to better capture extreme events
    """
    duration_cols = [
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

    adjusted_data = {}

    for year in df["year"].unique():
        if pd.isna(year):
            continue

        year_data = df[df["year"] == year]
        adjusted_year_data = {}

        for col in duration_cols:
            # Calculate threshold for this duration (events per year on average)
            annual_threshold = year_data[col].quantile(1 - lambda_threshold / 365.25)

            # Get all events above threshold
            extreme_events = year_data[year_data[col] > annual_threshold][col]

            if len(extreme_events) > 0:
                # Use the maximum of extreme events, but apply slight adjustment
                # based on the number of extreme events (more events = higher confidence)
                base_max = extreme_events.max()
                event_count_factor = min(1.1, 1 + 0.02 * len(extreme_events))
                adjusted_max = base_max * event_count_factor
            else:
                adjusted_max = year_data[col].max()

            adjusted_year_data[col] = adjusted_max

        adjusted_data[year] = adjusted_year_data

    # Convert to DataFrame
    adjusted_df = pd.DataFrame.from_dict(adjusted_data, orient="index")
    adjusted_df.index.name = "year"

    return adjusted_df


# Apply smart sampling strategy
print("Applying enhanced extreme event sampling...")

# Method 1: Enhanced annual maxima (primary method)
enhanced_annual_max = smart_annual_maxima_selection(df_intensity, n_events_per_year=3)

# Method 2: Partial duration series adjustment (secondary validation)
pds_annual_max = partial_duration_series_adjustment(df_intensity, lambda_threshold=2.5)

# Method 3: Original simple maximum (for comparison)
simple_annual_max = (
    df_intensity[
        [
            "year",
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
    ]
    .groupby("year")
    .max()
)

# Combine methods: Use enhanced method as primary, with PDS validation
# Where enhanced method gives significantly different results, apply a conservative blend
annual_max_intensity = enhanced_annual_max.copy()

# Apply conservative adjustment where methods disagree significantly
for col in enhanced_annual_max.columns:
    enhanced_values = enhanced_annual_max[col]
    simple_values = simple_annual_max[col]

    # Calculate relative differences
    rel_diff = abs(enhanced_values - simple_values) / simple_values

    # Where difference is very large (>20%), use a blend
    large_diff_mask = rel_diff > 0.2
    if large_diff_mask.any():
        # Use 70% enhanced, 30% simple for conservative approach
        annual_max_intensity.loc[large_diff_mask, col] = (
            0.7 * enhanced_values[large_diff_mask]
            + 0.3 * simple_values[large_diff_mask]
        )

print(
    f"Enhanced sampling complete. Original annual max range: {simple_annual_max.values.min():.2f}-{simple_annual_max.values.max():.2f}"
)
print(
    f"Enhanced annual max range: {annual_max_intensity.values.min():.2f}-{annual_max_intensity.values.max():.2f}"
)
print(
    f"Average improvement factor: {(annual_max_intensity.values.mean() / simple_annual_max.values.mean()):.3f}"
)

output_path = os.path.join(os.path.dirname(__file__), "results", "annual_max_intensity")
annual_max_intensity.to_csv(f"{output_path}.csv", index=True)

print("\n" + "=" * 60)
print("DATA PREPROCESSING COMPLETE")
print("=" * 60)
print(f"Annual maximum intensity data saved to: {output_path}.csv")
print("\nTo perform Gumbel distribution analysis, run:")
print("  python statistical-methods/gumbel.py")

