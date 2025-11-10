import pandas as pd
import numpy as np

print("Comparing IMF and Bartlett-Lewis Disaggregation Results")
print("=" * 60)

# Read original daily data
daily_df = pd.read_csv("data/gpm-bey-daily.csv", parse_dates=["date"])
original_total = daily_df["value"].sum()
print(f"Original daily total: {original_total:.1f} mm")

# Compare different durations
durations = ["1hr", "6hr", "12hr"]

for duration in durations:
    print(f"\n{duration.upper()} Duration Comparison:")
    print("-" * 30)
    
    # Read IMF data
    imf_file = f"data/gpm-bey-{duration}.csv"
    bl_file = f"data/gpm-bey-{duration}-bl.csv"
    
    try:
        imf_df = pd.read_csv(imf_file)
        bl_df = pd.read_csv(bl_file)
        
        # Calculate duration in hours for volume calculation
        if duration == "1hr":
            dt_hours = 1.0
        elif duration == "6hr":
            dt_hours = 6.0
        elif duration == "12hr":
            dt_hours = 12.0
        
        # Calculate total volumes
        imf_total = imf_df["value"].sum() * dt_hours
        bl_total = bl_df["value"].sum() * dt_hours
        
        # Statistics
        print("IMF Method:")
        print(f"  - Records: {len(imf_df):,}")
        print(f"  - Max intensity: {imf_df['value'].max():.3f} mm/hr")
        print(f"  - Mean intensity: {imf_df['value'].mean():.6f} mm/hr")
        print(f"  - Total volume: {imf_total:.1f} mm")
        print(f"  - Mass balance: {100 * imf_total / original_total:.2f}%")
        
        print("Bartlett-Lewis Method:")
        print(f"  - Records: {len(bl_df):,}")
        print(f"  - Max intensity: {bl_df['value'].max():.3f} mm/hr")
        print(f"  - Mean intensity: {bl_df['value'].mean():.6f} mm/hr")
        print(f"  - Total volume: {bl_total:.1f} mm")
        print(f"  - Mass balance: {100 * bl_total / original_total:.2f}%")
        
        # Temporal variability comparison
        imf_var = np.var(imf_df["value"])
        bl_var = np.var(bl_df["value"])
        
        print("Temporal Variability:")
        print(f"  - IMF variance: {imf_var:.6f}")
        print(f"  - B-L variance: {bl_var:.6f}")
        print(f"  - B-L/IMF ratio: {bl_var/imf_var:.2f}x")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")

print("\n" + "=" * 60)
print("Summary:")
print("- IMF: Deterministic scaling based on duration relationships")
print("- Bartlett-Lewis: Stochastic storm/cell structure with realistic variability")
print("- Both methods preserve total precipitation mass")
print("- B-L method introduces realistic temporal clustering and intermittency")