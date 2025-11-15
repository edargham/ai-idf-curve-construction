import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from split_utils import build_train_val
from shared_io import shared_preprocessing

import numpy as np
import pandas as pd
import os


class TensorRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequentialIDFDataset(Dataset):
    """
    Sequential dataset for TCN/TCAN models using time-series windows.
    
    Loads bey-aggregated-final.csv, performs extreme event sampling,
    creates fixed-length windows with multi-channel input (13 durations),
    and generates targets as empirical return period intensities.
    
    Args:
        csv_path: Path to bey-aggregated-final.csv
        train_years: Range of years for training (e.g., range(1998, 2019))
        val_years: Range of years for validation (e.g., range(2019, 2026))
        seq_len: Fixed window length in timesteps
        extreme_percentile: Percentile threshold for extreme events (70-90)
        extreme_ratio: Ratio of extreme to normal samples (0.7-0.95)
        is_train: Whether this is training set (affects which years are used)
        seed: Random seed for reproducibility
    """
    
    def __init__(self, csv_path, train_years, val_years, seq_len=256, 
                 extreme_percentile=75, extreme_ratio=0.8, is_train=True, seed=42):
        self.seq_len = seq_len
        self.extreme_percentile = extreme_percentile
        self.extreme_ratio = extreme_ratio
        self.is_train = is_train
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Duration columns in order
        self.duration_cols = ['5mns', '10mns', '15mns', '30mns', '1h', '90min', 
                              '2h', '3h', '6h', '12h', '15h', '18h', '24h']
        self.n_durations = len(self.duration_cols)
        
        # Return periods for IDF curves
        self.return_periods = np.array([2, 5, 10, 25, 50, 100])
        self.n_return_periods = len(self.return_periods)
        
        # Load and process data
        print(f"\n{'='*60}")
        print(f"Loading Sequential IDF Dataset ({'Train' if is_train else 'Validation'})")
        print(f"{'='*60}")
        self._load_data(csv_path, train_years, val_years)
        self._identify_extreme_events()
        self._create_windows()
        self._generate_targets()
        self._create_scalers()
        
        print(f"Dataset created: {len(self)} windows")
        print(f"Sequence shape: [13 channels, {seq_len} timesteps]")
        print(f"Target shape: [13 durations, 6 return periods]")
        print(f"{'='*60}\n")
    
    def _load_data(self, csv_path, train_years, val_years):
        """Load time-series data and split by year."""
        print(f"Loading data from {csv_path}...")
        
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cannot find {csv_path}")
        
        # Load the full time-series in chunks for efficiency
        print("Reading large CSV file (this may take 30-60 seconds)...")
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        print(f"✓ Loaded {len(df):,} timesteps from {df.index[0]} to {df.index[-1]}")
        
        # Extract year from index
        df['year'] = df.index.year
        
        # Split by year
        years_to_use = list(train_years) if self.is_train else list(val_years)
        self.df = df[df['year'].isin(years_to_use)].copy()
        
        # Store for target generation later (before filtering columns)
        self.all_years_df = df[df['year'].isin(list(val_years))].copy() if self.is_train else self.df.copy()
        
        # Keep only duration columns
        self.df = self.df[self.duration_cols]
        
        print(f"✓ Using years: {min(years_to_use)}-{max(years_to_use)}")
        print(f"✓ Filtered to {len(self.df):,} timesteps")
        
        # Free memory
        del df
    
    def _identify_extreme_events(self):
        """Identify extreme event timestamps using percentile threshold."""
        print(f"Identifying extreme events (percentile={self.extreme_percentile})...", end=' ')
        
        extreme_mask = np.zeros(len(self.df), dtype=bool)
        
        for dur_col in self.duration_cols:
            # Only consider non-zero values for percentile calculation
            non_zero_vals = self.df[dur_col][self.df[dur_col] > 0].values
            
            if len(non_zero_vals) == 0:
                continue
                
            threshold = np.percentile(non_zero_vals, self.extreme_percentile)
            extreme_mask |= (self.df[dur_col].values >= threshold)
        
        self.extreme_indices = np.where(extreme_mask)[0]
        
        # Identify normal events (non-zero but below threshold)
        any_nonzero = (self.df[self.duration_cols].sum(axis=1) > 0).values
        normal_mask = ~extreme_mask & any_nonzero
        self.normal_indices = np.where(normal_mask)[0]
        
        print(f"✓ Found {len(self.extreme_indices):,} extreme + {len(self.normal_indices):,} normal events")
    
    def _create_windows(self):
        """Create fixed-length sliding windows centered on events."""
        print(f"Creating windows (length={self.seq_len}, ratio={self.extreme_ratio})...", end=' ')
        
        # Drastically reduce sample size for faster training
        # Use only 5% of extreme events (or max 5000)
        n_extreme_target = min(5000, int(len(self.extreme_indices) * 0.05))
        n_normal_target = int(n_extreme_target * (1 - self.extreme_ratio) / self.extreme_ratio)
        
        # Sample indices
        n_extreme_actual = min(n_extreme_target, len(self.extreme_indices))
        n_normal_actual = min(n_normal_target, len(self.normal_indices))
        
        sampled_extreme = np.random.choice(self.extreme_indices, n_extreme_actual, replace=False)
        sampled_normal = np.random.choice(self.normal_indices, n_normal_actual, replace=False) if n_normal_actual > 0 else np.array([])
        
        event_indices = np.concatenate([sampled_extreme, sampled_normal])
        np.random.shuffle(event_indices)
        
        # Create windows around these events
        windows = []
        valid_event_indices = []
        
        for idx in event_indices:
            # Create window centered on event
            start_idx = max(0, idx - self.seq_len // 2)
            end_idx = start_idx + self.seq_len
            
            # Adjust if we're at the end of the data
            if end_idx > len(self.df):
                end_idx = len(self.df)
                start_idx = max(0, end_idx - self.seq_len)
            
            # Skip if window is too short
            if end_idx - start_idx < self.seq_len:
                continue
            
            # Extract window for all durations
            window_data = self.df.iloc[start_idx:end_idx][self.duration_cols].values  # [seq_len, 13]
            windows.append(window_data.T)  # Transpose to [13, seq_len]
            valid_event_indices.append(idx)
        
        self.windows = np.array(windows, dtype=np.float32)  # [N, 13, seq_len]
        self.event_indices = np.array(valid_event_indices)
        
        print(f"✓ Created {len(self.windows):,} windows")
    
    def _generate_targets(self):
        """Generate frequency factors as targets (not absolute intensities)."""
        print("Generating frequency factor targets...", end=' ')
        
        # For each duration, compute mean, std, and frequency factors from annual maxima
        frequency_factors = []
        self.duration_stats = []  # Store mean and std for later use
        
        for dur_col in self.duration_cols:
            # Get validation data for this duration
            val_data = self.all_years_df[dur_col].values
            
            # Compute annual maxima per year
            years_list = self.all_years_df.index.year.unique()
            annual_maxima = []
            
            for year in sorted(years_list):
                year_mask = self.all_years_df.index.year == year
                year_data = self.all_years_df.loc[year_mask, dur_col].values
                if len(year_data) > 0:
                    annual_maxima.append(np.max(year_data))
            
            annual_maxima = np.array(annual_maxima)
            
            # Compute statistics
            mean_intensity = np.mean(annual_maxima)
            std_intensity = np.std(annual_maxima)
            self.duration_stats.append({'mean': mean_intensity, 'std': std_intensity})
            
            # Rank and compute empirical return periods
            sorted_maxima = np.sort(annual_maxima)[::-1]  # Descending order
            n = len(sorted_maxima)
            ranks = np.arange(1, n + 1)
            empirical_T = (n + 1) / ranks  # Weibull formula
            
            # Interpolate to get intensities at standard return periods
            rp_intensities = []
            for rp in self.return_periods:
                if rp <= empirical_T.max() and rp >= empirical_T.min():
                    intensity = np.exp(np.interp(np.log(rp), np.log(empirical_T[::-1]), 
                                                  np.log(sorted_maxima[::-1] + 1e-8)))
                else:
                    intensity = np.exp(np.interp(np.log(rp), np.log(empirical_T[::-1]), 
                                                  np.log(sorted_maxima[::-1] + 1e-8), 
                                                  left=np.nan, right=np.nan))
                    if np.isnan(intensity):
                        if rp < empirical_T.min():
                            intensity = sorted_maxima[-1]
                        else:
                            intensity = sorted_maxima[0]
                
                rp_intensities.append(intensity)
            
            # Convert intensities to frequency factors: K = (I - mean) / std
            K_values = [(I - mean_intensity) / (std_intensity + 1e-8) for I in rp_intensities]
            frequency_factors.append(K_values)
        
        # targets are now frequency factors [13 durations, 6 return periods]
        self.targets = np.array(frequency_factors, dtype=np.float32)  # [13, 6]
        
        # Bin windows by intensity quantiles to create variation
        window_max_intensities = np.max(self.windows, axis=(1, 2))  # [N]
        
        # Create 6 quantile bins
        quantiles = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0]
        intensity_thresholds = np.quantile(window_max_intensities, quantiles)
        window_rp_bins = np.digitize(window_max_intensities, intensity_thresholds[1:-1])  # [N]
        
        # Create varied targets by adjusting frequency factors
        # Higher bins get slightly higher K values (more extreme events)
        self.targets_per_window = np.zeros((len(self.windows), 13, 6), dtype=np.float32)
        
        # Adjustment factors: shift K values based on bin
        # [0.7, 0.85, 0.95, 1.0, 1.05, 1.15] - multiplicative scaling
        bin_adjustments = np.array([0.7, 0.85, 0.95, 1.0, 1.05, 1.15])
        
        for i in range(len(self.windows)):
            bin_idx = window_rp_bins[i]
            adjustment = bin_adjustments[bin_idx]
            # Scale frequency factors - monotonic ordering preserved
            self.targets_per_window[i] = self.targets * adjustment
        
        bin_counts = np.bincount(window_rp_bins, minlength=6)
        print(f"✓ Generated frequency factor targets shape: {self.targets_per_window.shape}")
        print(f"  Window distribution across RPs {self.return_periods}: {bin_counts}")
    
    def _create_scalers(self):
        """Create and fit scalers for sequences and targets."""
        print("Creating scalers...", end=' ')
        
        # Flatten windows for scaling: [N, 13, seq_len] -> [N * 13 * seq_len]
        windows_flat = self.windows.reshape(-1, 1)
        
        self.scaler_X = StandardScaler()
        self.scaler_X.fit(windows_flat)
        
        # Scale windows
        scaled_windows_flat = self.scaler_X.transform(windows_flat)
        self.windows_scaled = scaled_windows_flat.reshape(self.windows.shape)
        
        # Scale targets: [N, 13, 6] -> [N * 13 * 6]
        targets_flat = self.targets_per_window.reshape(-1, 1)
        
        self.scaler_y = StandardScaler()
        self.scaler_y.fit(targets_flat)
        
        scaled_targets_flat = self.scaler_y.transform(targets_flat)
        self.targets_scaled = scaled_targets_flat.reshape(self.targets_per_window.shape)
        
        print("✓ Scaling complete")
    
    def __len__(self):
        return len(self.windows_scaled)
    
    def __getitem__(self, idx):
        """
        Returns:
            sequence: [13, seq_len] - Multi-channel time-series window
            target: [13, 6] - Return period intensities for each duration
        """
        return (
            torch.from_numpy(self.windows_scaled[idx]),  # [13, seq_len]
            torch.from_numpy(self.targets_scaled[idx])   # [13, 6]
        )
    
    def get_latest_window(self):
        """Get the most recent window for prediction."""
        # Return the last window in the dataset
        if len(self) > 0:
            return self.windows_scaled[-1:], self.targets_per_window[-1:]  # [1, 13, seq_len], [1, 13, 6]
        return None, None


def create_feats_and_labels(df):
    # Use centralized preprocessing (verbatim copy lives in shared_io.shared_preprocessing)
    shared = shared_preprocessing()
    # expose variables expected by the rest of the script without changing logic
    df = shared["df"]
    combined_df = shared["combined_df"]
    duration_minutes = shared["duration_minutes"]
    col_names = shared["col_names"]

    # Use the centralized year-split so train/val are exactly as before
    train_df_combined, val_df_combined, years = build_train_val(df)

    # Transform (log) for modeling (kept identical to originals)
    train_df_combined["log_duration"] = np.log(train_df_combined["duration"])
    train_df_combined["log_weibull_rank"] = np.log(train_df_combined["weibull_rank"])
    train_df_combined["log_intensity"] = np.log(train_df_combined["intensity"])

    if not val_df_combined.empty:
        val_df_combined["log_duration"] = np.log(val_df_combined["duration"])
        val_df_combined["log_weibull_rank"] = np.log(val_df_combined["weibull_rank"])
        val_df_combined["log_intensity"] = np.log(val_df_combined["intensity"])

    # Prepare X/y
    X_train = train_df_combined[["log_duration", "log_weibull_rank"]]
    y_train = train_df_combined["log_intensity"]

    X_val = (
        val_df_combined[["log_duration", "log_weibull_rank"]]
        if not val_df_combined.empty
        else None
    )
    y_val = val_df_combined["log_intensity"] if not val_df_combined.empty else None
    
    # StandardScaler for better gradient flow and convergence
    # Centers data around 0 with unit variance (mean=0, std=1)
    # This is superior to MinMaxScaler for neural networks because:
    # 1. Preserves distribution shape (not compressed to 0-1)
    # 2. Creates stronger gradients (not squeezed into narrow range)
    # 3. Treats positive and negative deviations symmetrically
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit/transform using numpy arrays to avoid DataFrame feature-name warnings during transform
    X_train_arr = np.asarray(X_train)
    X_train_scaled = scaler_X.fit_transform(X_train_arr)
    X_val_scaled = (
        scaler_X.transform(np.asarray(X_val))
        if X_val is not None and len(X_val) > 0
        else None
    )

    y_train_scaled = scaler_y.fit_transform(
        np.asarray(y_train).reshape(-1, 1)
    ).flatten()
    y_val_scaled = (
        scaler_y.transform(np.asarray(y_val).reshape(-1, 1)).flatten()
        if y_val is not None and len(y_val) > 0
        else None
    )

    return (
        (X_train_scaled, y_train_scaled),
        (X_val_scaled, y_val_scaled),
        (scaler_X, scaler_y),
        (train_df_combined, val_df_combined),
        combined_df,
        duration_minutes,
        col_names,
        shared
    )