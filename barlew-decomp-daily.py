import os
import pandas as pd
import numpy as np
from scipy.stats import gamma
import warnings
warnings.filterwarnings('ignore')

# Read the daily CSV file
file_path = os.path.join(
    os.path.dirname(__file__), "data", "gpm-bey-daily.csv"
)
df = pd.read_csv(file_path, parse_dates=["date"])


class BartlettLewisModel:
    """
    Bartlett-Lewis Rectangular Pulse Model for temporal rainfall disaggregation.
    
    The model represents rainfall as a series of rectangular pulses (storms) where:
    - Storm arrivals follow a Poisson process with rate λ
    - Each storm duration follows an exponential distribution with rate η
    - Cell arrivals within storms follow a Poisson process with rate β
    - Cell durations follow an exponential distribution with rate γ
    - Cell intensities follow a gamma distribution with parameters (α, θ)
    """
    
    def __init__(self):
        # Model parameters (will be fitted from data)
        self.lambda_param = 0.1    # Storm arrival rate (storms/hour)
        self.eta = 2.0             # Storm duration rate (1/hours)
        self.beta = 5.0            # Cell arrival rate within storms (cells/hour)
        self.gamma = 10.0          # Cell duration rate (1/hours)
        self.alpha = 2.0           # Gamma shape parameter for intensities
        self.theta = 1.0           # Gamma scale parameter for intensities
        
    def fit_parameters(self, daily_precip, n_wet_days):
        """
        Fit model parameters using method of moments and maximum likelihood.
        
        Parameters:
        daily_precip: Array of daily precipitation values (mm/day)
        n_wet_days: Number of wet days in the dataset
        """
        # Filter out dry days
        wet_precip = daily_precip[daily_precip > 0.1]  # Threshold for wet days
        
        if len(wet_precip) == 0:
            print("Warning: No wet days found, using default parameters")
            return
            
        # Basic statistics
        mean_precip = np.mean(wet_precip)
        var_precip = np.var(wet_precip)
        
        # Estimate parameters using simplified moment matching
        # These are empirical relationships based on rainfall characteristics
        
        # Storm arrival rate (storms per day)
        self.lambda_param = n_wet_days / len(daily_precip) * 24  # Convert to hourly rate
        
        # Storm duration (hours) - empirical estimate
        self.eta = 24.0 / (mean_precip / 2.0 + 1.0)  # Inverse of mean storm duration
        
        # Cell parameters - based on sub-storm structure
        self.beta = self.lambda_param * 3.0  # More cells than storms
        self.gamma = self.eta * 2.0          # Shorter cell durations
        
        # Intensity distribution parameters
        if var_precip > 0:
            # Method of moments for gamma distribution
            self.theta = var_precip / mean_precip
            self.alpha = mean_precip / self.theta
        else:
            self.alpha = 2.0
            self.theta = mean_precip / 2.0
            
        # Ensure parameters are positive and reasonable
        self.lambda_param = max(0.01, min(self.lambda_param, 2.0))
        self.eta = max(0.1, min(self.eta, 10.0))
        self.beta = max(0.1, min(self.beta, 20.0))
        self.gamma = max(0.5, min(self.gamma, 50.0))
        self.alpha = max(0.5, min(self.alpha, 10.0))
        self.theta = max(0.1, min(self.theta, 20.0))
        
        print("Fitted parameters:")
        print(f"  λ (storm rate): {self.lambda_param:.3f} /hour")
        print(f"  η (storm duration rate): {self.eta:.3f} /hour")
        print(f"  β (cell rate): {self.beta:.3f} /hour")
        print(f"  γ (cell duration rate): {self.gamma:.3f} /hour")
        print(f"  α (intensity shape): {self.alpha:.3f}")
        print(f"  θ (intensity scale): {self.theta:.3f}")
    
    def generate_rainfall_day(self, daily_total, target_duration_min, random_state=None):
        """
        Generate disaggregated rainfall for a single day using the B-L model.
        
        Parameters:
        daily_total: Total daily precipitation (mm)
        target_duration_min: Target temporal resolution in minutes
        random_state: Random seed for reproducibility
        
        Returns:
        Array of rainfall intensities for each interval (mm/hr)
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # If no rain, return zeros
        if daily_total <= 0.01:
            intervals_per_day = int(24 * 60 / target_duration_min)
            return np.zeros(intervals_per_day)
        
        # Time resolution in hours
        dt = target_duration_min / 60.0
        time_steps = int(24 / dt)
        rainfall_intensity = np.zeros(time_steps)
        
        # Generate storms for the day
        # Expected number of storms in 24 hours
        n_storms_expected = self.lambda_param * 24
        n_storms = np.random.poisson(n_storms_expected)
        
        if n_storms == 0:
            # If no storms generated, distribute rainfall uniformly (fallback)
            avg_intensity = daily_total / 24.0  # mm/hr
            rainfall_intensity[:] = avg_intensity
            return rainfall_intensity
        
        total_generated = 0.0
        
        for storm_idx in range(n_storms):
            # Storm arrival time (uniform over 24 hours)
            storm_start = np.random.uniform(0, 24)
            
            # Storm duration
            storm_duration = np.random.exponential(1.0 / self.eta)
            storm_end = min(storm_start + storm_duration, 24)
            
            if storm_end <= storm_start:
                continue
                
            # Generate cells within this storm
            storm_length = storm_end - storm_start
            n_cells_expected = self.beta * storm_length
            n_cells = max(1, np.random.poisson(n_cells_expected))
            
            for cell_idx in range(n_cells):
                # Cell start time within storm
                cell_start = storm_start + np.random.uniform(0, storm_length * 0.8)
                
                # Cell duration
                cell_duration = np.random.exponential(1.0 / self.gamma)
                cell_end = min(cell_start + cell_duration, storm_end)
                
                if cell_end <= cell_start:
                    continue
                
                # Cell intensity
                cell_intensity = gamma.rvs(a=self.alpha, scale=self.theta)
                
                # Add cell to time series
                start_idx = int(cell_start / dt)
                end_idx = int(cell_end / dt)
                
                if start_idx < time_steps and end_idx > start_idx:
                    end_idx = min(end_idx, time_steps)
                    rainfall_intensity[start_idx:end_idx] += cell_intensity
                    total_generated += cell_intensity * (end_idx - start_idx) * dt
        
        # Scale to match daily total (mass conservation)
        if total_generated > 0:
            scaling_factor = daily_total / total_generated
            rainfall_intensity *= scaling_factor
        else:
            # Fallback: uniform distribution
            avg_intensity = daily_total / 24.0
            rainfall_intensity[:] = avg_intensity
            
        return rainfall_intensity
    
    def disaggregate_daily_series(self, dates, daily_precip, target_duration_min):
        """
        Disaggregate entire daily precipitation series.
        
        Parameters:
        dates: Array of daily dates
        daily_precip: Array of daily precipitation values (mm/day)
        target_duration_min: Target temporal resolution in minutes
        
        Returns:
        tuple: (disaggregated_dates, disaggregated_intensities)
        """
        intervals_per_day = int(24 * 60 / target_duration_min)
        n_days = len(dates)
        total_intervals = n_days * intervals_per_day
        
        # Pre-allocate output arrays
        new_dates = np.zeros(total_intervals, dtype="datetime64[ns]")
        new_intensities = np.zeros(total_intervals, dtype=np.float64)
        
        print(f"Disaggregating to {target_duration_min}-minute intervals...")
        print(f"Processing {n_days} days...")
        
        for day_idx in range(n_days):
            if day_idx % 500 == 0:
                print(f"  Progress: {day_idx}/{n_days} days ({100*day_idx/n_days:.1f}%)")
                
            daily_total = daily_precip[day_idx]
            
            # Generate disaggregated rainfall for this day
            day_intensities = self.generate_rainfall_day(
                daily_total, target_duration_min, 
                random_state=42 + day_idx  # Reproducible randomness
            )
            
            # Generate timestamps for this day
            base_date = dates[day_idx]
            day_dates = base_date + np.arange(intervals_per_day) * np.timedelta64(target_duration_min, 'm')
            
            # Store in output arrays
            start_idx = day_idx * intervals_per_day
            end_idx = start_idx + intervals_per_day
            new_dates[start_idx:end_idx] = day_dates
            new_intensities[start_idx:end_idx] = day_intensities
        
        print(f"  Completed: {n_days}/{n_days} days (100.0%)")
        return new_dates, new_intensities


def create_aggregated_dataset_bl(dates, daily_precip, target_duration_hours):
    """
    Create aggregated dataset for durations longer than 24 hours using simple averaging.
    
    Parameters:
    dates: Array of daily dates
    daily_precip: Array of daily precipitation values (mm/day)
    target_duration_hours: Target duration in hours
    
    Returns:
    tuple: (dates_array, values_array) for the aggregated data
    """
    # Convert daily precipitation to daily intensity (mm/hr)
    daily_intensities = daily_precip / 24.0
    
    # For durations longer than 24 hours, use simple temporal averaging
    # This maintains the same daily resolution but represents longer-duration averages
    target_intensities = daily_intensities * (24.0 / target_duration_hours)
    
    return dates, target_intensities


# Extract dates and precipitation values
dates = df["date"].values
daily_precip_values = df["value"].values.astype(np.float64)

# Count wet days for parameter fitting
wet_days = np.sum(daily_precip_values > 0.1)  # Threshold of 0.1 mm for wet days

print("Starting Bartlett-Lewis disaggregation from daily data...")
print(f"Processing {len(df)} days of data")
print(f"Wet days: {wet_days} ({100*wet_days/len(df):.1f}%)")

# Initialize and fit the Bartlett-Lewis model
bl_model = BartlettLewisModel()
bl_model.fit_parameters(daily_precip_values, wet_days)

# Define all target durations and their properties
# Format: (duration_name, duration_minutes, filename)
disaggregation_configs = [
    ("5min", 5, "gpm-bey-5mns.csv"),
    ("10min", 10, "gpm-bey-10mns.csv"),
    ("15min", 15, "gpm-bey-15mns.csv"),
    ("30min", 30, "gpm-bey-30mns.csv"),
    ("1hr", 60, "gpm-bey-1hr.csv"),
    ("90min", 90, "gpm-bey-90min.csv"),
    ("2hr", 120, "gpm-bey-2hr.csv"),
    ("3hr", 180, "gpm-bey-3hr.csv"),
    ("6hr", 360, "gpm-bey-6hr.csv"),
    ("12hr", 720, "gpm-bey-12hr.csv"),
    ("15hr", 15, "gpm-bey-15hr.csv"),
    ("18hr", 18, "gpm-bey-18hr.csv"),
]

# Define aggregation configurations (durations longer than 24 hours)
# Format: (duration_name, duration_hours, filename)
aggregation_configs = [
]

# Process disaggregated datasets (durations ≤ 12 hours)
print("\nProcessing disaggregated datasets using Bartlett-Lewis model...")
for duration_name, duration_min, filename in disaggregation_configs:
    print(f"\nCreating {duration_name} dataset...")
    
    new_dates, new_values = bl_model.disaggregate_daily_series(
        dates, daily_precip_values, duration_min
    )
    
    # Create DataFrame and save
    df_new = pd.DataFrame({"date": new_dates, "value": new_values})
    output_path = os.path.join("./data", filename.replace(".csv", "-bl.csv"))
    df_new.to_csv(output_path, index=False)
    
    print(f"  - Shape: {df_new.shape}")
    print(f"  - Max intensity: {df_new['value'].max():.3f} mm/hr")
    print(f"  - Mean intensity: {df_new['value'].mean():.3f} mm/hr")
    print(f"  - Total volume: {df_new['value'].sum() * duration_min / 60:.1f} mm")
    print(f"  - Original daily total: {daily_precip_values.sum():.1f} mm")
    print(f"  - Saved to: {filename.replace('.csv', '-bl.csv')}")

# Process aggregated datasets (durations between 12-24 hours)
print("\nProcessing aggregated datasets...")
for duration_name, duration_hours, filename in aggregation_configs:
    print(f"Creating {duration_name} dataset...")
    
    new_dates, new_values = create_aggregated_dataset_bl(
        dates, daily_precip_values, duration_hours
    )
    
    # Create DataFrame and save
    df_new = pd.DataFrame({"date": new_dates, "value": new_values})
    output_path = os.path.join("./data", filename.replace(".csv", "-bl.csv"))
    df_new.to_csv(output_path, index=False)
    
    print(f"  - Shape: {df_new.shape}")
    print(f"  - Mean intensity: {df_new['value'].mean():.3f} mm/hr")
    print(f"  - Saved to: {filename.replace('.csv', '-bl.csv')}")

# Also create the daily dataset (as intensity rather than precipitation)
print("\nCreating daily intensity dataset...")
daily_intensities = daily_precip_values / 24.0
df_daily_intensity = pd.DataFrame({"date": dates, "value": daily_intensities})
df_daily_intensity.to_csv("./data/gpm-bey-daily-intensity-bl.csv", index=False)
print(f"  - Shape: {df_daily_intensity.shape}")
print("  - Saved to: gpm-bey-daily-intensity-bl.csv")

print("\n" + "="*70)
print("BARTLETT-LEWIS DISAGGREGATION COMPLETE")
print("="*70)
print("All datasets generated from 24-hour (daily) data using Bartlett-Lewis Rectangular Pulse Model")
print(f"Original daily data: {df.shape[0]} days")
print(f"Wet days processed: {wet_days} days ({100*wet_days/len(df):.1f}%)")

print("\nGenerated datasets (with '-bl' suffix):")

print("\nDisaggregated datasets (sub-daily):")
for duration_name, duration_min, filename in disaggregation_configs:
    intervals_per_day = int(24 * 60 / duration_min)
    total_intervals = len(df) * intervals_per_day
    print(f"  - {duration_name:>6}: {total_intervals:>8} intervals ({filename.replace('.csv', '-bl.csv')})")

print("\nAggregated datasets (sub-daily to daily):")
for duration_name, duration_hours, filename in aggregation_configs:
    print(f"  - {duration_name:>6}: {len(df):>8} intervals ({filename.replace('.csv', '-bl.csv')})")

print(f"\nDaily intensity: {len(df):>8} intervals (gpm-bey-daily-intensity-bl.csv)")

print("\nAll files saved to ./data/ directory with '-bl' suffix")
print("\nModel characteristics:")
print("- Stochastic temporal disaggregation using rectangular pulses")
print("- Storm and cell structure preserved")
print("- Mass conservation enforced")
print("- Parameters fitted to local precipitation statistics")
