import os
import pandas as pd
import numpy as np
from scipy.stats import gamma
import warnings
warnings.filterwarnings('ignore')

# Read the 30-minute CSV file
file_path = os.path.join(
    os.path.dirname(__file__), "data", "gpm-bey-30mns.csv"
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
        
    def fit_parameters(self, precip_values, time_interval_hours, n_wet_intervals):
        """
        Fit model parameters using method of moments and maximum likelihood.
        
        Parameters:
        precip_values: Array of precipitation values (mm/interval)
        time_interval_hours: Duration of each interval in hours
        n_wet_intervals: Number of wet intervals in the dataset
        """
        # Convert to intensity (mm/hr)
        intensities = precip_values / time_interval_hours
        
        # Filter out dry intervals
        wet_intensities = intensities[intensities > 0.1]  # Threshold for wet intervals
        
        if len(wet_intensities) == 0:
            print("Warning: No wet intervals found, using default parameters")
            return
            
        # Basic statistics
        mean_intensity = np.mean(wet_intensities)
        var_intensity = np.var(wet_intensities)
        
        # Estimate parameters using simplified moment matching
        # These are empirical relationships based on rainfall characteristics
        
        # Storm arrival rate (storms per hour)
        wet_fraction = n_wet_intervals / len(precip_values)
        self.lambda_param = wet_fraction * (1.0 / time_interval_hours) * 2.0
        
        # Storm duration (hours) - empirical estimate
        self.eta = 1.0 / (time_interval_hours * 2.0)  # Inverse of mean storm duration
        
        # Cell parameters - based on sub-storm structure
        self.beta = self.lambda_param * 3.0  # More cells than storms
        self.gamma = self.eta * 2.0          # Shorter cell durations
        
        # Intensity distribution parameters
        if var_intensity > 0:
            # Method of moments for gamma distribution
            self.theta = var_intensity / mean_intensity
            self.alpha = mean_intensity / self.theta
        else:
            self.alpha = 2.0
            self.theta = mean_intensity / 2.0
            
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
    
    def generate_rainfall_interval(self, interval_total, source_duration_hours, 
                                   target_duration_min, random_state=None):
        """
        Generate disaggregated rainfall for a single interval using the B-L model.
        
        Parameters:
        interval_total: Total precipitation in the interval (mm)
        source_duration_hours: Duration of the source interval in hours
        target_duration_min: Target temporal resolution in minutes
        random_state: Random seed for reproducibility
        
        Returns:
        Array of rainfall intensities for each sub-interval (mm/hr)
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # If no rain, return zeros
        if interval_total <= 0.01:
            n_subintervals = int(source_duration_hours * 60 / target_duration_min)
            return np.zeros(n_subintervals)
        
        # Time resolution in hours
        dt = target_duration_min / 60.0
        time_steps = int(source_duration_hours / dt)
        rainfall_intensity = np.zeros(time_steps)
        
        # Generate storms for the interval
        # Expected number of storms
        n_storms_expected = self.lambda_param * source_duration_hours
        n_storms = max(1, np.random.poisson(n_storms_expected))
        
        total_generated = 0.0
        
        for storm_idx in range(n_storms):
            # Storm arrival time (uniform over interval)
            storm_start = np.random.uniform(0, source_duration_hours)
            
            # Storm duration
            storm_duration = np.random.exponential(1.0 / self.eta)
            storm_end = min(storm_start + storm_duration, source_duration_hours)
            
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
        
        # Scale to match interval total (mass conservation)
        if total_generated > 0:
            scaling_factor = interval_total / total_generated
            rainfall_intensity *= scaling_factor
        else:
            # Fallback: uniform distribution
            avg_intensity = interval_total / source_duration_hours
            rainfall_intensity[:] = avg_intensity
            
        return rainfall_intensity
    
    def disaggregate_series(self, dates, precip_values, source_duration_min, target_duration_min):
        """
        Disaggregate precipitation series to finer temporal resolution.
        
        Parameters:
        dates: Array of timestamps
        precip_values: Array of precipitation values (mm/interval)
        source_duration_min: Duration of source intervals in minutes
        target_duration_min: Target temporal resolution in minutes
        
        Returns:
        tuple: (disaggregated_dates, disaggregated_intensities)
        """
        source_duration_hours = source_duration_min / 60.0
        n_subintervals = int(source_duration_min / target_duration_min)
        n_intervals = len(dates)
        total_subintervals = n_intervals * n_subintervals
        
        # Pre-allocate output arrays
        new_dates = np.zeros(total_subintervals, dtype="datetime64[ns]")
        new_intensities = np.zeros(total_subintervals, dtype=np.float64)
        
        print(f"Disaggregating from {source_duration_min}min to {target_duration_min}min intervals...")
        print(f"Processing {n_intervals} intervals...")
        
        for interval_idx in range(n_intervals):
            if interval_idx % 5000 == 0:
                print(f"  Progress: {interval_idx}/{n_intervals} intervals ({100*interval_idx/n_intervals:.1f}%)")
                
            interval_total = precip_values[interval_idx]
            
            # Generate disaggregated rainfall for this interval
            interval_intensities = self.generate_rainfall_interval(
                interval_total, source_duration_hours, target_duration_min,
                random_state=42 + interval_idx  # Reproducible randomness
            )
            
            # Generate timestamps for this interval
            base_date = dates[interval_idx]
            interval_dates = base_date + np.arange(n_subintervals) * np.timedelta64(target_duration_min, 'm')
            
            # Store in output arrays
            start_idx = interval_idx * n_subintervals
            end_idx = start_idx + n_subintervals
            new_dates[start_idx:end_idx] = interval_dates
            new_intensities[start_idx:end_idx] = interval_intensities
        
        print(f"  Completed: {n_intervals}/{n_intervals} intervals (100.0%)")
        return new_dates, new_intensities


def aggregate_data(dates, values, source_duration_min, target_duration_min):
    """
    Aggregate precipitation data to coarser temporal resolution.
    
    Parameters:
    dates: Array of timestamps
    values: Array of precipitation intensity values (mm/hr)
    source_duration_min: Duration of source intervals in minutes
    target_duration_min: Target temporal resolution in minutes (must be multiple of source)
    
    Returns:
    tuple: (aggregated_dates, aggregated_intensities)
    """
    if target_duration_min % source_duration_min != 0:
        raise ValueError("Target duration must be a multiple of source duration")
    
    n_intervals_to_combine = int(target_duration_min / source_duration_min)
    n_source = len(dates)
    n_target = n_source // n_intervals_to_combine
    
    # Pre-allocate output arrays
    new_dates = np.zeros(n_target, dtype="datetime64[ns]")
    new_intensities = np.zeros(n_target, dtype=np.float64)
    
    print(f"Aggregating from {source_duration_min}min to {target_duration_min}min...")
    print(f"Combining {n_intervals_to_combine} intervals into each aggregated interval...")
    
    for i in range(n_target):
        start_idx = i * n_intervals_to_combine
        end_idx = start_idx + n_intervals_to_combine
        
        # Use the first timestamp of the aggregated period
        new_dates[i] = dates[start_idx]
        
        # Average the intensities (they're already in mm/hr)
        new_intensities[i] = np.mean(values[start_idx:end_idx])
    
    print(f"  Completed: Created {n_target} aggregated intervals")
    return new_dates, new_intensities


# Extract dates and precipitation values
dates = df["date"].values
precip_values = df["value"].values.astype(np.float64)

# Convert to intensity (mm/hr) - source data is 30-minute intervals
source_duration_min = 30
source_duration_hours = source_duration_min / 60.0
intensities = precip_values / source_duration_hours

# Count wet intervals for parameter fitting
wet_intervals = np.sum(precip_values > 0.01)  # Threshold of 0.01 mm for wet intervals

print("Starting Bartlett-Lewis processing from 30-minute data...")
print(f"Processing {len(df)} intervals ({len(df) * source_duration_hours / 24:.1f} days)")
print(f"Wet intervals: {wet_intervals} ({100*wet_intervals/len(df):.1f}%)")

# Initialize and fit the Bartlett-Lewis model
bl_model = BartlettLewisModel()
bl_model.fit_parameters(precip_values, source_duration_hours, wet_intervals)

# Define disaggregation configurations (30min → finer resolutions)
# Format: (duration_name, duration_minutes, filename)
disaggregation_configs = [
    ("5min", 5, "gpm-bey-5mns-bl.csv"),
    ("10min", 10, "gpm-bey-10mns-bl.csv"),
    ("15min", 15, "gpm-bey-15mns-bl.csv"),
]

# Define aggregation configurations (30min → coarser resolutions)
# Format: (duration_name, duration_minutes, filename)
aggregation_configs = [
    ("1hr", 60, "gpm-bey-1hr-bl.csv"),
    ("90min", 90, "gpm-bey-90min-bl.csv"),
    ("2hr", 120, "gpm-bey-2hr-bl.csv"),
    ("3hr", 180, "gpm-bey-3hr-bl.csv"),
    ("6hr", 360, "gpm-bey-6hr-bl.csv"),
    ("12hr", 720, "gpm-bey-12hr-bl.csv"),
    ("15hr", 900, "gpm-bey-15hr-bl.csv"),
    ("18hr", 1080, "gpm-bey-18hr-bl.csv"),
]

# Process disaggregated datasets (30min → finer)
print("\n" + "="*70)
print("DISAGGREGATION: 30min → finer resolutions")
print("="*70)
for duration_name, duration_min, filename in disaggregation_configs:
    print(f"\nCreating {duration_name} dataset...")
    
    new_dates, new_values = bl_model.disaggregate_series(
        dates, precip_values, source_duration_min, duration_min
    )
    
    # Create DataFrame and save
    df_new = pd.DataFrame({"date": new_dates, "value": new_values})
    output_path = os.path.join("./data", filename)
    df_new.to_csv(output_path, index=False)
    
    print(f"  - Shape: {df_new.shape}")
    print(f"  - Max intensity: {df_new['value'].max():.3f} mm/hr")
    print(f"  - Mean intensity: {df_new['value'].mean():.3f} mm/hr")
    print(f"  - Total volume: {df_new['value'].sum() * duration_min / 60:.1f} mm")
    print(f"  - Original total: {precip_values.sum():.1f} mm")
    print(f"  - Saved to: {filename}")

# Process aggregated datasets (30min → coarser)
print("\n" + "="*70)
print("AGGREGATION: 30min → coarser resolutions")
print("="*70)
for duration_name, duration_min, filename in aggregation_configs:
    print(f"\nCreating {duration_name} dataset...")
    
    new_dates, new_values = aggregate_data(
        dates, intensities, source_duration_min, duration_min
    )
    
    # Create DataFrame and save
    df_new = pd.DataFrame({"date": new_dates, "value": new_values})
    output_path = os.path.join("./data", filename)
    df_new.to_csv(output_path, index=False)
    
    print(f"  - Shape: {df_new.shape}")
    print(f"  - Max intensity: {df_new['value'].max():.3f} mm/hr")
    print(f"  - Mean intensity: {df_new['value'].mean():.3f} mm/hr")
    print(f"  - Saved to: {filename}")

print("\n" + "="*70)
print("BARTLETT-LEWIS PROCESSING COMPLETE")
print("="*70)
print("All datasets generated from 30-minute data")
print(f"Source data: {df.shape[0]} intervals ({len(df) * source_duration_hours / 24:.1f} days)")
print(f"Wet intervals: {wet_intervals} ({100*wet_intervals/len(df):.1f}%)")

print("\nGenerated datasets:")
print("\nDisaggregated (30min → finer):")
for duration_name, duration_min, filename in disaggregation_configs:
    n_subintervals = source_duration_min / duration_min
    total_intervals = int(len(df) * n_subintervals)
    print(f"  - {duration_name:>6}: {total_intervals:>8} intervals ({filename})")

print("\nAggregated (30min → coarser):")
for duration_name, duration_min, filename in aggregation_configs:
    n_combined = duration_min / source_duration_min
    total_intervals = int(len(df) / n_combined)
    print(f"  - {duration_name:>6}: {total_intervals:>8} intervals ({filename})")

print("\nAll files saved to ./data/ directory with '-bl' suffix")
print("\nModel characteristics:")
print("- Disaggregation: Stochastic temporal disaggregation using rectangular pulses")
print("- Aggregation: Simple temporal averaging of intensities")
print("- Mass conservation enforced for disaggregation")
print("- Parameters fitted to 30-minute precipitation statistics")
