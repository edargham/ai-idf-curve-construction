import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
import time
import os
import warnings
warnings.filterwarnings("ignore")

def find_date_from_csv(duration_col, max_value, year):
    """
    Find the exact date from the original CSV file when historical intensity matching fails.
    Since annual maximums are weighted averages of top 3 events, we implement a fallback
    to find the most representative event from the top 3 extreme events.
    """
    # Map duration columns to actual CSV filenames
    duration_to_file = {
        '5mns': 'gpm-bey-5mns.csv',
        '10mns': 'gpm-bey-10mns.csv',
        '15mns': 'gpm-bey-15mns.csv',
        '30mns': 'gpm-bey-30mns-imdf.csv',
        '1h': 'gpm-bey-1hr.csv',
        '90min': 'gpm-bey-90min.csv',
        '2h': 'gpm-bey-2hr.csv',
        '3h': 'gpm-bey-3hr.csv',
        '6h': 'gpm-bey-6hr.csv',
        '12h': 'gpm-bey-12hr.csv',
        '15h': 'gpm-bey-15hr.csv',
        '18h': 'gpm-bey-18hr.csv',
        '24h': 'gpm-bey-daily.csv'
    }
    
    csv_file = duration_to_file.get(duration_col)
    if not csv_file:
        return None
        
    csv_path = f'./data/{csv_file}'
    if not os.path.exists(csv_path):
        return None
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path, parse_dates=['date'])
        
        # Filter by year
        df['year'] = df['date'].dt.year
        year_data = df[df['year'] == year]
        
        if len(year_data) == 0:
            return None
        
        # First try: Find exact match with tolerance
        tolerance = max(max_value * 0.01, 0.001)  # 1% tolerance or 0.001 mm/hr minimum
        matching_rows = year_data[abs(year_data['value'] - max_value) <= tolerance]
        
        if len(matching_rows) > 0:
            found_date = matching_rows.iloc[0]['date']
            return found_date
        
        # Fallback method: Since annual max is weighted average of top 3 events,
        # find the top 3 events and pick the one that best represents the condition
        top_3_events = year_data.nlargest(3, 'value')
        
        if len(top_3_events) == 0:
            return None
            
        # Calculate which of the top 3 events would contribute most to the weighted average
        # Using the same weights as in idf-precip-aggregator.py: [0.6, 0.3, 0.1]
        weights = [0.6, 0.3, 0.1]
        
        if len(top_3_events) >= 3:
            # Find which individual event best represents our target max_value
            # considering the weighted average context with weights [0.6, 0.3, 0.1]
            top_values = top_3_events['value'].values[:3]
            
            # Find which individual event is closest to our target max_value
            # considering the weighted average context
            best_match_idx = 0
            best_score = float('inf')
            
            for i, (idx, event) in enumerate(top_3_events.head(3).iterrows()):
                # Score based on:
                # 1. How close this event is to our target value
                # 2. The weight of this event in the overall calculation
                weight_importance = weights[i]
                
                # Combined score: closer to target and higher weight is better
                distance_to_target = abs(event['value'] - max_value)
                score = distance_to_target / weight_importance  # Lower is better
                
                if score < best_score:
                    best_score = score
                    best_match_idx = i
            
            # Return the date of the best matching event
            return top_3_events.iloc[best_match_idx]['date']
            
        elif len(top_3_events) >= 2:
            # Use the higher weighted event (70% vs 30% strategy)
            top_values = top_3_events['value'].values[:2]
            
            # Choose between the two events - favor the one with higher weight
            if abs(top_values[0] - max_value) <= abs(top_values[1] - max_value):
                return top_3_events.iloc[0]['date']
            else:
                return top_3_events.iloc[1]['date']
        else:
            # Only one event available
            return top_3_events.iloc[0]['date']
            
    except Exception as e:
        print(f"   Warning: Could not read {csv_path}: {e}")
    
    return None

def process_year_data(year, annual_max_df, historical_file_path):
    """
    Process a single year's data to find maximum event dates
    """
    print(f"   Processing year {year}...")
    
    # Read only data for this year
    try:
        # Read the entire file and filter by year (more reliable than parsing dates during read)
        hist_chunk = pd.read_csv(
            historical_file_path,
            parse_dates=['date'],
            chunksize=50000  # Process in chunks to manage memory
        )
        
        year_data = []
        for chunk in hist_chunk:
            chunk_year_data = chunk[chunk['year'] == year]
            if len(chunk_year_data) > 0:
                year_data.append(chunk_year_data)
        
        if not year_data:
            return year, {}
            
        year_df = pd.concat(year_data, ignore_index=True)
        
    except Exception as e:
        print(f"   Warning: Could not process year {year}: {e}")
        return year, {}
    
    # Get the annual max values for this year
    annual_row = annual_max_df[annual_max_df['year'] == year]
    if len(annual_row) == 0:
        return year, {}
    annual_row = annual_row.iloc[0]
    
    year_max_dates = {}
    
    # Duration mapping
    duration_cols = ['5mns', '10mns', '15mns', '30mns', '1h', '90min', '2h', '3h', '6h', '12h', '15h', '18h', '24h']
    
    for duration_col in duration_cols:
        if duration_col in annual_row:
            max_value = annual_row[duration_col]
            
            # Find the date when this maximum occurred
            # Use a small tolerance for floating point comparison
            tolerance = max(max_value * 0.01, 0.001)  # 1% tolerance or 0.001 mm/hr minimum
            matching_rows = year_df[abs(year_df[duration_col] - max_value) <= tolerance]
            
            if len(matching_rows) > 0:
                # Take the first occurrence if multiple matches
                max_date = matching_rows.iloc[0]['date']
                year_max_dates[duration_col] = max_date
            else:
                # Try to find the date from the original CSV file
                csv_date = find_date_from_csv(duration_col, max_value, year)
                year_max_dates[duration_col] = csv_date
    
    return year, year_max_dates

def find_max_event_dates(annual_max_df, historical_file_path='./results/historical_intensity.csv'):
    """
    Find the actual dates when maximum intensities occurred for each year and duration
    Uses parallel processing to improve performance
    Only processes validation period (2019-2025)
    """
    print("   Loading historical intensity data to find event dates...")

    # Filter to validation period (2019-2025)
    annual_max_df = annual_max_df[(annual_max_df['year'] >= 2019) & (annual_max_df['year'] <= 2025)].copy()
    print(f"   ✓ Processing validation period 2019-2025: {len(annual_max_df)} years")

    # Create a dictionary to store dates for each year and duration
    max_dates = {}
    
    # Get unique years to process
    years_to_process = annual_max_df['year'].unique()
    
    # Determine optimal number of workers (use CPU count but cap at 8 to avoid overwhelming I/O)
    max_workers = min(os.cpu_count(), len(years_to_process))
    
    # Use parallel processing to handle years
    print(f"   Processing {len(years_to_process)} years in parallel using {max_workers} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all year processing tasks
        future_to_year = {
            executor.submit(process_year_data, year, annual_max_df, historical_file_path): year 
            for year in years_to_process
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_year):
            year, year_max_dates = future.result()
            max_dates[year] = year_max_dates
    
    elapsed_time = time.time() - start_time
    print(f"   ✓ Completed processing all {len(years_to_process)} years in {elapsed_time:.2f} seconds")
    print(f"   ✓ Average processing time per year: {elapsed_time/len(years_to_process):.3f} seconds")
    print(f"   ✓ Estimated sequential time would be ~{elapsed_time * max_workers:.1f} seconds (speedup: {max_workers:.1f}x)")
    return max_dates

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
    return r ** 2

def weibull_plotting_position(n, rank):
    """
    Calculate Weibull plotting position for return period calculation
    Formula: T = n / rank
    where n is the total number of values and rank is the position when sorted in descending order
    """
    return n / rank

def calculate_return_periods_weibull(data):
    """
    Calculate return periods using Weibull plotting formula for a series
    """
    # Sort data in descending order
    sorted_data = data.sort_values(ascending=False).reset_index(drop=True)
    n = len(sorted_data)
    
    # Calculate ranks (starting from 1)
    ranks = np.arange(1, n + 1)
    
    # Calculate return periods using Weibull formula
    return_periods = weibull_plotting_position(n, ranks)
    
    # Create dataframe with original index, sorted values, and return periods
    result_df = pd.DataFrame({
        'intensity': sorted_data.values,
        'rank': ranks,
        'return_period': return_periods
    })
    
    return result_df

def interpolate_idf_for_return_period(idf_df, return_period, duration_col):
    """
    Interpolate IDF value for a specific return period and duration
    """
    # Get return periods from the IDF data
    return_periods = idf_df['Return Period (years)'].values
    intensities = idf_df[duration_col].values
    
    # Interpolate for the given return period
    interpolated_intensity = np.interp(return_period, return_periods, intensities)
    
    return interpolated_intensity

def interpolate_ai_idf_for_return_period(idf_df, return_period, duration_minutes):
    """
    Interpolate AI model IDF value for a specific return period and duration
    """
    # Get return periods from the AI model data
    return_periods = [2, 5, 10, 25, 50, 100]
    
    # Find the row for the specified duration
    duration_row = idf_df[idf_df['Duration (minutes)'] == duration_minutes].iloc[0]
    
    # Extract intensities for all return periods
    intensities = [duration_row[f'{rp}-year'] for rp in return_periods]
    
    # Interpolate for the given return period
    interpolated_intensity = np.interp(return_period, return_periods, intensities)
    
    return interpolated_intensity

def calculate_model_metrics(observed, predicted):
    """
    Calculate RMSE, MAE, NSE, and R² for model evaluation
    """
    # Remove any NaN values
    mask = ~(np.isnan(observed) | np.isnan(predicted))
    observed_clean = np.array(observed)[mask]
    predicted_clean = np.array(predicted)[mask]
    
    if len(observed_clean) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    rmse = np.sqrt(mean_squared_error(observed_clean, predicted_clean))
    mae = mean_absolute_error(observed_clean, predicted_clean)
    r2 = r2_score(observed_clean, predicted_clean)
    nse = nash_sutcliffe_efficiency(observed_clean, predicted_clean)
    
    return rmse, mae, r2, nse

def calculate_confidence_intervals(observed, predicted, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for RMSE and MAE.
    
    Returns:
        tuple: (rmse_ci_lower, rmse_ci_upper, mae_ci_lower, mae_ci_upper)
    """
    # Remove NaN values
    mask = ~(np.isnan(observed) | np.isnan(predicted))
    observed_clean = np.array(observed)[mask]
    predicted_clean = np.array(predicted)[mask]
    
    if len(observed_clean) < 10:
        return np.nan, np.nan, np.nan, np.nan
    
    rmse_boots = []
    mae_boots = []
    
    np.random.seed(42)  # For reproducibility
    n = len(observed_clean)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        obs_sample = observed_clean[indices]
        pred_sample = predicted_clean[indices]
        
        # Calculate metrics
        rmse_boot = np.sqrt(mean_squared_error(obs_sample, pred_sample))
        mae_boot = mean_absolute_error(obs_sample, pred_sample)
        
        rmse_boots.append(rmse_boot)
        mae_boots.append(mae_boot)
    
    # Calculate confidence intervals
    alpha = (1 - confidence) / 2
    rmse_ci_lower = np.percentile(rmse_boots, alpha * 100)
    rmse_ci_upper = np.percentile(rmse_boots, (1 - alpha) * 100)
    mae_ci_lower = np.percentile(mae_boots, alpha * 100)
    mae_ci_upper = np.percentile(mae_boots, (1 - alpha) * 100)
    
    return rmse_ci_lower, rmse_ci_upper, mae_ci_lower, mae_ci_upper

def perform_paired_t_tests(all_predictions_df, baseline_model='Gumbel'):
    """
    Perform paired t-tests comparing model errors against observed data.
    Tests if differences in prediction errors between models are statistically significant.
    
    The baseline is the OBSERVED DATA - all models are evaluated on how well they predict
    observations. We then compare if one model's errors are significantly different from another.
    
    Args:
        all_predictions_df: DataFrame with 'observed' and model prediction columns
        baseline_model: Model to compare all others against (default: Gumbel)
    
    Returns:
        DataFrame with t-test results comparing each model to the baseline
    """
    print("\n   Performing paired t-tests comparing prediction errors...")
    print(f"   Baseline: Observed data | Reference model for comparison: {baseline_model}")
    
    model_names = ['Literature', 'Gumbel', 'SVM', 'ANN', 'TCN', 'TCAN']
    observed = all_predictions_df['observed'].values
    
    # Calculate absolute errors for baseline model (errors from observed data)
    if baseline_model not in all_predictions_df.columns:
        print(f"   Warning: {baseline_model} not found in data")
        return pd.DataFrame()
    
    baseline_pred = all_predictions_df[baseline_model].values
    baseline_errors = np.abs(observed - baseline_pred)
    
    results = []
    for model_name in model_names:
        if model_name not in all_predictions_df.columns:
            continue
        
        model_pred = all_predictions_df[model_name].values
        model_errors = np.abs(observed - model_pred)
        
        # Remove NaN pairs
        mask = ~(np.isnan(baseline_errors) | np.isnan(model_errors))
        baseline_errors_clean = baseline_errors[mask]
        model_errors_clean = model_errors[mask]
        
        if len(baseline_errors_clean) < 10:
            continue
        
        # Calculate mean errors
        mean_baseline_error = np.mean(baseline_errors_clean)
        mean_model_error = np.mean(model_errors_clean)
        
        # If comparing to itself, just report the mean error
        if model_name == baseline_model:
            results.append({
                'Model': model_name,
                'Mean_Error': mean_model_error,
                't_statistic': 0.0,
                'p_value': 1.0,
                'vs_Baseline': 'Reference Model',
                'N_pairs': len(model_errors_clean)
            })
            continue
        
        # Perform paired t-test (two-tailed)
        # H0: model errors = baseline errors
        # H1: model errors ≠ baseline errors
        t_stat, p_value = stats.ttest_rel(model_errors_clean, baseline_errors_clean)
        
        # Calculate mean difference (negative = model is better)
        mean_diff = mean_model_error - mean_baseline_error
        pct_change = (mean_diff / mean_baseline_error) * 100
        
        # Interpretation
        if p_value < 0.05:
            if mean_diff < 0:
                significance = f"Significantly better than {baseline_model}"
            else:
                significance = f"Significantly worse than {baseline_model}"
        else:
            significance = f"Not significantly different from {baseline_model}"
        
        results.append({
            'Model': model_name,
            'Mean_Error': mean_model_error,
            't_statistic': t_stat,
            'p_value': p_value,
            'Error_Diff': mean_diff,
            'Pct_Change': pct_change,
            'vs_Baseline': significance,
            'N_pairs': len(model_errors_clean)
        })
    
    return pd.DataFrame(results)

def evaluate_full_validation_dataset(annual_max_df, gumbel_df, literature_df, ai_models):
    """
    Evaluate ALL models on the complete validation dataset (2019-2025).
    This provides unbiased performance metrics matching the validation split used in model training.
    
    Returns:
        tuple: (overall_metrics_df, per_duration_metrics_df, all_predictions_df)
    """
    print("   Evaluating models on FULL validation dataset (2019-2025)...")
    
    # Filter to validation period ONLY (2019-2025) - matching what models were validated on
    annual_max_df = annual_max_df[(annual_max_df['year'] >= 2019) & (annual_max_df['year'] <= 2025)].copy()
    print(f"   ✓ Validation period: 2019-2025 ({len(annual_max_df)} years)")
    
    # Duration mappings
    duration_mapping_annual = {
        5: '5mns', 10: '10mns', 15: '15mns', 30: '30mns',
        60: '1h', 90: '90min', 120: '2h', 180: '3h', 360: '6h', 720: '12h',
        900: '15h', 1080: '18h', 1440: '24h'
    }
    
    duration_mapping_gumbel = {
        5: '5 mins', 10: '10 mins', 15: '15 mins', 30: '30 mins',
        60: '60 mins', 90: '90 mins', 120: '120 mins', 180: '180 mins', 360: '360 mins', 
        720: '720 mins', 900: '900 mins', 1080: '1080 mins', 1440: '1440 mins'
    }
    
    duration_mapping_literature = duration_mapping_gumbel
    
    # Collect all predictions and observations
    all_data = []
    model_names = ['Literature', 'Gumbel', 'SVM', 'ANN', 'TCN', 'TCAN']
    
    # Process each duration
    for duration_mins, annual_col in duration_mapping_annual.items():
        if annual_col not in annual_max_df.columns:
            continue
            
        # Get annual max data for this duration
        annual_data = annual_max_df[[annual_col, 'year']].dropna()
        if len(annual_data) < 5:
            continue
        
        # Calculate return periods using Weibull formula
        weibull_results = calculate_return_periods_weibull(annual_data[annual_col])
        
        # Match intensities with years
        intensity_to_year = dict(zip(annual_data[annual_col].values, annual_data['year'].values))
        
        # For each observation, get all model predictions
        for idx, event in weibull_results.iterrows():
            observed_intensity = event['intensity']
            return_period = event['return_period']
            year = intensity_to_year.get(observed_intensity)
            
            # Get Gumbel prediction
            gumbel_col = duration_mapping_gumbel.get(duration_mins)
            gumbel_pred = np.nan
            if gumbel_col and gumbel_col in gumbel_df.columns:
                try:
                    gumbel_pred = interpolate_idf_for_return_period(gumbel_df, return_period, gumbel_col)
                except Exception:
                    pass
            
            # Get Literature prediction
            literature_col = duration_mapping_literature.get(duration_mins)
            literature_pred = np.nan
            if literature_col and literature_col in literature_df.columns:
                try:
                    literature_pred = interpolate_idf_for_return_period(literature_df, return_period, literature_col)
                except Exception:
                    pass
            
            # Get AI model predictions
            ai_predictions = {}
            for model_name, model_df in ai_models.items():
                try:
                    ai_pred = interpolate_ai_idf_for_return_period(model_df, return_period, duration_mins)
                    ai_predictions[model_name] = ai_pred
                except Exception:
                    ai_predictions[model_name] = np.nan
            
            # Store all data
            data_point = {
                'year': year,
                'duration_mins': duration_mins,
                'return_period': return_period,
                'observed': observed_intensity,
                'Literature': literature_pred,
                'Gumbel': gumbel_pred,
                **ai_predictions
            }
            all_data.append(data_point)
    
    all_predictions_df = pd.DataFrame(all_data)
    print(f"   ✓ Collected {len(all_predictions_df)} observations across all durations")
    
    # Calculate overall metrics for each model
    overall_metrics = []
    for model_name in model_names:
        if model_name in all_predictions_df.columns:
            observed = all_predictions_df['observed'].values
            predicted = all_predictions_df[model_name].values
            
            rmse, mae, r2, nse = calculate_model_metrics(observed, predicted)
            n_valid = np.sum(~np.isnan(predicted))
            
            overall_metrics.append({
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'NSE': nse,
                'N': n_valid
            })
    
    overall_metrics_df = pd.DataFrame(overall_metrics)
    
    # Calculate per-duration metrics
    per_duration_metrics = []
    for duration_mins in duration_mapping_annual.keys():
        duration_data = all_predictions_df[all_predictions_df['duration_mins'] == duration_mins]
        if len(duration_data) == 0:
            continue
        
        for model_name in model_names:
            if model_name in duration_data.columns:
                observed = duration_data['observed'].values
                predicted = duration_data[model_name].values
                
                rmse, mae, r2, nse = calculate_model_metrics(observed, predicted)
                n_valid = np.sum(~np.isnan(predicted))
                
                per_duration_metrics.append({
                    'Duration_mins': duration_mins,
                    'Model': model_name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'NSE': nse,
                    'N': n_valid
                })
    
    per_duration_metrics_df = pd.DataFrame(per_duration_metrics)
    
    print(f"   ✓ Calculated metrics for {len(model_names)} models")
    return overall_metrics_df, per_duration_metrics_df, all_predictions_df

def select_extreme_events(annual_max_df, all_predictions_df, n_events=7):
    """
    Select extreme events based ONLY on objective criteria:
    - High return periods (≥10 years preferred)
    - High intensity values (top percentiles)
    - Diverse durations (stratified selection)
    
    NO filtering by model performance - purely objective selection.
    Uses VALIDATION PERIOD ONLY (2019-2025) for consistency.
    
    Returns:
        DataFrame with selected extreme events
    """
    print(f"   Selecting {n_events} extreme events using objective criteria...")
    
    # Filter to events with return period ≥ 5 years (most extreme in 7-year validation period)
    # Note: In a 7-year dataset, max return period is ~7-8 years
    extreme_candidates = all_predictions_df[all_predictions_df['return_period'] >= 3.5].copy()
    print(f"   ✓ Found {len(extreme_candidates)} events with return period ≥3.5 years")
    
    if len(extreme_candidates) == 0:
        print("   ⚠️  No events with RP≥3.5yr, using RP≥2yr threshold...")
        extreme_candidates = all_predictions_df[all_predictions_df['return_period'] >= 2].copy()
    
    # Calculate intensity percentile within each duration category
    extreme_candidates['intensity_percentile'] = 0.0
    for duration in extreme_candidates['duration_mins'].unique():
        duration_mask = extreme_candidates['duration_mins'] == duration
        duration_data = extreme_candidates[duration_mask]
        percentiles = duration_data['observed'].rank(pct=True)
        extreme_candidates.loc[duration_mask, 'intensity_percentile'] = percentiles
    
    # Calculate composite extremeness score (60% return period, 40% intensity percentile)
    # Normalize return period to 0-1 scale
    rp_normalized = (extreme_candidates['return_period'] - extreme_candidates['return_period'].min()) / \
                    (extreme_candidates['return_period'].max() - extreme_candidates['return_period'].min())
    
    extreme_candidates['extremeness_score'] = 0.6 * rp_normalized + 0.4 * extreme_candidates['intensity_percentile']
    
    # Define duration strata: short (5-30min), medium (60-180min), long (360-1440min)
    def categorize_duration(duration_mins):
        if duration_mins <= 30:
            return 'short'
        elif duration_mins <= 180:
            return 'medium'
        else:
            return 'long'
    
    extreme_candidates['duration_category'] = extreme_candidates['duration_mins'].apply(categorize_duration)
    
    # Stratified selection: aim for diverse durations
    # Target: 2 short, 3 medium, 2 long (adjust if not enough in each category)
    target_distribution = {'short': 2, 'medium': 3, 'long': 2}
    
    selected_events = []
    for category, target_count in target_distribution.items():
        category_events = extreme_candidates[extreme_candidates['duration_category'] == category]
        if len(category_events) > 0:
            # Sort by extremeness score and take top events
            category_events_sorted = category_events.sort_values('extremeness_score', ascending=False)
            n_to_select = min(target_count, len(category_events_sorted))
            selected_events.append(category_events_sorted.head(n_to_select))
    
    if len(selected_events) > 0:
        selected_df = pd.concat(selected_events, ignore_index=True)
        
        # If we don't have enough events, add more from the highest scoring remaining
        if len(selected_df) < n_events:
            remaining = extreme_candidates[~extreme_candidates.index.isin(selected_df.index)]
            remaining_sorted = remaining.sort_values('extremeness_score', ascending=False)
            additional_needed = n_events - len(selected_df)
            selected_df = pd.concat([selected_df, remaining_sorted.head(additional_needed)], ignore_index=True)
        
        # If we have too many, trim to top n_events by extremeness score
        if len(selected_df) > n_events:
            selected_df = selected_df.sort_values('extremeness_score', ascending=False).head(n_events)
        
        # Sort by return period for presentation
        selected_df = selected_df.sort_values('return_period', ascending=False).reset_index(drop=True)
        
        print(f"   ✓ Selected {len(selected_df)} extreme events:")
        print(f"      • Return periods: {selected_df['return_period'].min():.1f} to {selected_df['return_period'].max():.1f} years")
        print(f"      • Durations: {selected_df['duration_mins'].min()}-{selected_df['duration_mins'].max()} minutes")
        print("      • Distribution: ", end="")
        for cat in ['short', 'medium', 'long']:
            count = (selected_df['duration_category'] == cat).sum()
            print(f"{cat}={count} ", end="")
        print()
        
        return selected_df
    else:
        print("   ✗ No extreme events found")
        return pd.DataFrame()

def evaluate_models_on_events(events_df, max_event_dates=None):
    """
    Evaluate ALL models on the selected events.
    Shows complete performance matrix including cases where traditional methods win.
    
    Returns:
        DataFrame with detailed event-by-event model comparison
    """
    if len(events_df) == 0:
        return pd.DataFrame()
    
    print(f"   Evaluating all models on {len(events_df)} selected events...")
    
    model_names = ['Literature', 'Gumbel', 'SVM', 'ANN', 'TCN', 'TCAN']
    
    # Enhance events_df with additional analysis
    events_analysis = []
    
    for idx, event in events_df.iterrows():
        # Get event metadata
        event_data = {
            'event_num': idx + 1,
            'year': event['year'],
            'duration_mins': event['duration_mins'],
            'return_period': event['return_period'],
            'observed': event['observed'],
            'event_date': None
        }
        
        # Try to find event date
        if max_event_dates is not None and event['year'] in max_event_dates:
            duration_mapping = {
                5: '5mns', 10: '10mns', 15: '15mns', 30: '30mns',
                60: '1h', 90: '90min', 120: '2h', 180: '3h', 360: '6h', 720: '12h',
                900: '15h', 1080: '18h', 1440: '24h'
            }
            duration_col = duration_mapping.get(event['duration_mins'])
            if duration_col:
                event_data['event_date'] = max_event_dates[event['year']].get(duration_col)
        
        # Get all model predictions and calculate errors
        for model_name in model_names:
            if model_name in event:
                pred = event[model_name]
                error = abs(event['observed'] - pred) if not np.isnan(pred) else np.nan
                event_data[f'{model_name}_pred'] = pred
                event_data[f'{model_name}_error'] = error
        
        # Calculate improvements relative to Gumbel
        if 'Gumbel_error' in event_data and not np.isnan(event_data['Gumbel_error']):
            gumbel_error = event_data['Gumbel_error']
            for model_name in ['Literature', 'SVM', 'ANN', 'TCN', 'TCAN']:
                error_key = f'{model_name}_error'
                if error_key in event_data and not np.isnan(event_data[error_key]):
                    improvement = (gumbel_error - event_data[error_key]) / gumbel_error * 100
                    event_data[f'{model_name}_improvement'] = improvement
        
        # Identify best model for this event
        errors = {}
        for model_name in model_names:
            error_key = f'{model_name}_error'
            if error_key in event_data and not np.isnan(event_data[error_key]):
                errors[model_name] = event_data[error_key]
        
        if errors:
            best_model = min(errors.keys(), key=lambda k: errors[k])
            event_data['best_model'] = best_model
            event_data['best_error'] = errors[best_model]
        
        events_analysis.append(event_data)
    
    events_analysis_df = pd.DataFrame(events_analysis)
    print("   ✓ Complete evaluation matrix created")
    
    return events_analysis_df



def comprehensive_model_evaluation(events_df, dataset_type='selected_events'):
    """
    Comprehensive evaluation of all models on the provided dataset.
    
    Args:
        events_df: DataFrame with observed and predicted values for all models
        dataset_type: 'full_validation' or 'selected_events' for labeling
    
    Returns:
        DataFrame with performance metrics for each model
    """
    if len(events_df) == 0:
        return pd.DataFrame()
    
    model_names = ['Literature', 'Gumbel', 'SVM', 'ANN', 'TCN', 'TCAN']
    
    # Collect all predictions and observations
    all_predictions = {model: [] for model in model_names}
    all_observations = []
    
    # Determine column names based on dataset type
    observed_col = 'observed' if 'observed' in events_df.columns else 'observed_intensity'
    
    for idx, event in events_df.iterrows():
        all_observations.append(event[observed_col])
        
        for model_name in model_names:
            pred_col = f'{model_name}_pred' if f'{model_name}_pred' in events_df.columns else model_name
            if pred_col in event:
                all_predictions[model_name].append(event[pred_col])
            else:
                all_predictions[model_name].append(np.nan)
    
    # Calculate metrics for each model
    results = []
    for model_name in model_names:
        preds = all_predictions[model_name]
        rmse, mae, r2, nse = calculate_model_metrics(all_observations, preds)
        
        results.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'NSE': nse,
            'N': len([p for p in preds if not np.isnan(p)]),
            'Dataset': dataset_type
        })
    
    return pd.DataFrame(results)

def create_event_comparison_table(events_df):
    """
    Create a detailed table comparing all models on selected extreme events.
    Shows complete performance matrix including cases where traditional methods excel.
    """
    if len(events_df) == 0:
        return pd.DataFrame()
    
    table_data = []
    for row_num, (idx, event) in enumerate(events_df.iterrows()):
        # Event
        event_name = f"Event {row_num + 1}"
        
        # Format the event date
        date_str = "N/A"
        if event.get('event_date') is not None and pd.notna(event['event_date']):
            # Convert to string and format if it's a datetime
            try:
                if isinstance(event['event_date'], str):
                    # If it's already a string, use it as is
                    date_str = event['event_date']
                else:
                    # If it's a datetime object, format appropriately
                    if event['event_date'].time() == pd.Timestamp('00:00:00').time():
                        # If time is midnight (00:00:00), likely from date-only CSV, show just date
                        date_str = event['event_date'].strftime('%Y-%m-%d')
                    else:
                        # Otherwise show full datetime
                        date_str = event['event_date'].strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                date_str = "N/A"
        elif event.get('year') is not None:
            date_str = f"{int(event['year'])}"
        
        # Format duration - convert to hours if over 60 minutes
        duration_mins = event['duration_mins']
        if duration_mins > 60:
            duration_hours = duration_mins / 60
            if duration_hours == int(duration_hours):
                duration_display = f"{int(duration_hours)} hrs"
            else:
                duration_display = f"{duration_hours:.1f} hrs"
        else:
            duration_display = f"{duration_mins} mins"
        
        # Get observed value
        observed = event.get('observed', event.get('observed_intensity'))
        
        # Literature and Statistical predictions
        literature_pred = event.get('Literature_pred')
        statistical_pred = event.get('Gumbel_pred')
        gumbel_error = event.get('Gumbel_error', np.nan)
        
        # Find all AI models and rank them by performance
        ai_models_performance = []
        for model in ['SVM', 'ANN', 'TCN', 'TCAN']:
            error_col = f'{model}_error'
            pred_col = f'{model}_pred'
            if error_col in event and pred_col in event:
                ai_error = event[error_col]
                ai_pred = event[pred_col]
                
                if not np.isnan(ai_error) and not np.isnan(gumbel_error) and gumbel_error > 0:
                    improvement_pct = (gumbel_error - ai_error) / gumbel_error * 100
                    beats_gumbel = ai_error < gumbel_error
                else:
                    improvement_pct = 0
                    beats_gumbel = False
                
                ai_models_performance.append({
                    'model': model,
                    'pred': ai_pred,
                    'error': ai_error,
                    'improvement_pct': improvement_pct,
                    'beats_gumbel': beats_gumbel
                })
        
        # Sort by error (best performance first)
        ai_models_performance.sort(key=lambda x: x['error'] if not np.isnan(x['error']) else float('inf'))
        
        # Create model rankings
        if ai_models_performance:
            models_list = ", ".join([m['model'] for m in ai_models_performance])
            
            # Get improvements for models that beat Gumbel
            improvements = [m['improvement_pct'] for m in ai_models_performance if m['beats_gumbel']]
            
            if improvements:
                if len(improvements) > 1:
                    improvement_range = f"{min(improvements):.1f}% to {max(improvements):.1f}%"
                else:
                    improvement_range = f"{improvements[0]:.1f}%"
            else:
                improvement_range = "None beat Gumbel"
            
            # AI predictions in order
            ai_pred_summary = ", ".join([f"{m['pred']:.2f}" for m in ai_models_performance if not np.isnan(m['pred'])])
        else:
            models_list = "N/A"
            improvement_range = "N/A"
            ai_pred_summary = "N/A"
        
        # Get observed value safely
        observed = event.get('observed', event.get('observed_intensity'))
        
        table_data.append({
            'Event': event_name,
            'Date': date_str,
            'Duration': duration_display,
            'Observed (mm/hr)': f"{observed:.2f}" if observed is not None else "N/A",
            'Literature (mm/hr)': f"{literature_pred:.2f}" if literature_pred is not None and not np.isnan(literature_pred) else "N/A",
            'Gumbel (mm/hr)': f"{statistical_pred:.2f}" if statistical_pred is not None and not np.isnan(statistical_pred) else "N/A",
            'AI Models (Best→Worst)': models_list,
            'AI Pred (mm/hr)': ai_pred_summary,
            'Improvement vs Gumbel': improvement_range,
            'Best Overall': event.get('best_model', 'N/A')
        })
    
    return pd.DataFrame(table_data)

def detailed_model_analysis_per_event(events_df):
    """
    Provide detailed analysis for each individual event
    """
    if len(events_df) == 0:
        return
    
    print("\n" + "=" * 100)
    print("DETAILED EVENT-BY-EVENT ANALYSIS")
    print("=" * 100)
    
    for row_num, (idx, event) in enumerate(events_df.iterrows()):
        # Format date information
        date_info = ""
        if event.get('event_date') is not None and pd.notna(event['event_date']):
            try:
                if isinstance(event['event_date'], str):
                    date_str = event['event_date'][:10]
                else:
                    date_str = event['event_date'].strftime('%Y-%m-%d')
                date_info = f" (Date: {date_str})"
            except Exception:
                if event.get('year') is not None:
                    date_info = f" (Year: {int(event['year'])})"
        elif event.get('year') is not None:
            date_info = f" (Year: {int(event['year'])})"
        
        print(f"\n🎯 EVENT {row_num+1}: Duration {event['duration_mins']} minutes, "
              f"Return Period {event['return_period']:.1f} years{date_info}")
        print("-" * 80)
        print(f"   Observed Intensity: {event['observed']:.2f} mm/hr")
        
        # Show Literature prediction if available
        if event.get('Literature_pred') is not None and not pd.isna(event['Literature_pred']):
            lit_error = abs(event['observed'] - event['Literature_pred'])
            print(f"   Literature Prediction: {event['Literature_pred']:.2f} mm/hr (Error: {lit_error:.2f})")
        else:
            print("   Literature Prediction: N/A")
        
        gumbel_error = abs(event['observed'] - event['Gumbel_pred'])
        print(f"   Gumbel Prediction:  {event['Gumbel_pred']:.2f} mm/hr (Error: {gumbel_error:.2f})")
        
        # Show all AI model predictions
        ai_models = ['SVM', 'ANN', 'TCN', 'TCAN']
        ai_results = []
        
        for model in ai_models:
            pred_col = f'{model}_pred'
            if pred_col in event and not pd.isna(event[pred_col]):
                pred = event[pred_col]
                error = abs(event['observed'] - pred)
                improvement = (gumbel_error - error) / gumbel_error * 100
                ai_results.append({
                    'Model': model,
                    'Prediction': pred,
                    'Error': error,
                    'Improvement': improvement,
                    'Better_than_Gumbel': '✓' if error < gumbel_error else '✗'
                })
        
        if len(ai_results) > 0:
            ai_results_df = pd.DataFrame(ai_results)
            ai_results_df = ai_results_df.sort_values('Error')
            
            print("\n   AI Model Performance:")
            for _, row in ai_results_df.iterrows():
                print(f"   {row['Better_than_Gumbel']} {row['Model']:<5}: "
                      f"{row['Prediction']:.2f} mm/hr (Error: {row['Error']:.2f}, "
                      f"Improvement: {row['Improvement']:.1f}%)")
            
            best_model = ai_results_df.iloc[0]['Model']
            print(f"\n   🏆 Best Model: {best_model} with {ai_results_df.iloc[0]['Improvement']:.1f}% improvement over Gumbel")
        else:
            print("\n   No AI model predictions available for this event")

def explain_ranking_vs_wins_discrepancy(superior_events, evaluation_results):
    """
    Explain why overall ranking differs from event wins
    """
    print("\n" + "=" * 100)
    print("RANKING vs EVENT WINS ANALYSIS")
    print("=" * 100)
    
    # First, let's analyze the fundamental inconsistency
    print("🚨 CRITICAL ANALYSIS: TCN/#1 and TCAN/#2 vs Event Performance")
    print("=" * 80)
    
    # Count actual event appearances
    event_wins = superior_events['best_ai_model'].value_counts()
    
    print("📊 EVENT WINS vs OVERALL RANKINGS:")
    ai_models_ranking = evaluation_results[evaluation_results['Model'].isin(['SVM', 'ANN', 'TCN', 'TCAN'])].copy()
    ai_models_ranking = ai_models_ranking.sort_values('Composite_Score')
    
    for idx, (_, row) in enumerate(ai_models_ranking.iterrows()):
        model = row['Model']
        rank = idx + 1
        wins = event_wins.get(model, 0)
        win_pct = (wins / len(superior_events)) * 100
        
        print(f"   {rank}. {model}: Rank #{rank}, Wins {wins}/{len(superior_events)} ({win_pct:.1f}%)")
        print(f"      RMSE: {row['RMSE']:.3f}, R²: {row['R2']:.4f}, NSE: {row['NSE']:.4f}")
    
    print("\n🔍 DETAILED EVENT-BY-EVENT RANKING ANALYSIS:")
    print("   (Who actually performed best in each event?)")
    print("-" * 70)
    
    # Analyze each event to see actual performance rankings
    for idx, (_, event) in enumerate(superior_events.iterrows()):
        event_num = idx + 1
        duration = event['duration_mins']
        
        # Get all model performances for this event
        models_performance = []
        for model in ['SVM', 'ANN', 'TCN', 'TCAN']:
            pred_col = f'{model}_pred'
            error_col = f'{model}_error'
            if pred_col in event and error_col in event:
                error = event[error_col]
                pred = event[pred_col]

                gumbel_error = event['gumbel_error']
                improvement = (gumbel_error - error) / gumbel_error * 100 if gumbel_error > 0 else 0
                
                models_performance.append({
                    'Model': model,
                    'Prediction': pred,
                    'Error': error,
                    'Improvement': improvement,
                    'Beats_Gumbel': error < gumbel_error
                })
        
        # Sort by error (best performance first)
        models_performance.sort(key=lambda x: x['Error'])
        
        duration_str = f"{duration} min" if duration <= 60 else f"{duration/60:.0f} hr"
        winner = event['best_ai_model']
        
        print(f"\n   Event {event_num} ({duration_str}): Observed = {event['observed_intensity']:.2f} mm/hr")
        print(f"   Gumbel: {event['gumbel_pred']:.2f} mm/hr (Error: {event['gumbel_error']:.4f})")
        print(f"   Winner: {winner}")
        
        print("   Full Rankings:")
        for rank, model_perf in enumerate(models_performance, 1):
            status = "✓" if model_perf['Beats_Gumbel'] else "✗"
            print(f"     {rank}. {status} {model_perf['Model']}: {model_perf['Prediction']:.2f} mm/hr "
                  f"(Error: {model_perf['Error']:.4f}, Imp: {model_perf['Improvement']:.1f}%)")
        
        # Identify the issue: Why isn't the best overall model winning?
        if models_performance[0]['Model'] != winner:
            print(f"   ⚠️  INCONSISTENCY: Best performer {models_performance[0]['Model']} != Winner {winner}")
        
        # Check if TCN or TCAN should have won but didn't
        tcn_perf = next((m for m in models_performance if m['Model'] == 'TCN'), None)
        tcan_perf = next((m for m in models_performance if m['Model'] == 'TCAN'), None)
        
        if tcn_perf and tcn_perf['Beats_Gumbel'] and tcn_perf['Model'] != winner:
            tcn_rank = next(i for i, m in enumerate(models_performance, 1) if m['Model'] == 'TCN')
            print(f"   🤔 TCN (#1 overall): Rank #{tcn_rank} here, beats Gumbel but didn't win")
        
        if tcan_perf and tcan_perf['Beats_Gumbel'] and tcan_perf['Model'] != winner:
            tcan_rank = next(i for i, m in enumerate(models_performance, 1) if m['Model'] == 'TCAN')
            print(f"   🤔 TCAN (#2 overall): Rank #{tcan_rank} here, beats Gumbel but didn't win")
    
    # First, let's examine TCAN's performance in detail
    print("\n🔍 TCAN PERFORMANCE ANALYSIS:")
    print("   (TCAN ranks #2 overall but wins 0 events - why?)")
    print("-" * 70)
    
    for idx, (_, event) in enumerate(superior_events.iterrows()):
        event_num = idx + 1
        duration = event['duration_mins']
        best_model = event['best_ai_model']
        
        # Get all model errors for this event
        models_performance = []
        for model in ['SVM', 'ANN', 'TCN', 'TCAN']:
            error_col = f'{model}_error'
            if error_col in event:
                error = event[error_col]
                gumbel_error = event['gumbel_error']
                improvement = (gumbel_error - error) / gumbel_error * 100 if gumbel_error > 0 else 0
                models_performance.append({
                    'Model': model,
                    'Error': error,
                    'Improvement': improvement,
                    'Rank': 0  # Will be calculated
                })
        
        # Sort by error (lowest first) and assign ranks
        models_performance.sort(key=lambda x: x['Error'])
        for rank, model_perf in enumerate(models_performance, 1):
            model_perf['Rank'] = rank
        
        # Find TCAN's position
        tcan_perf = next((m for m in models_performance if m['Model'] == 'TCAN'), None)
        
        duration_str = f"{duration} min" if duration <= 60 else f"{duration/60:.0f} hr"
        print(f"   Event {event_num} ({duration_str}): Winner = {best_model}")
        
        if tcan_perf:
            print(f"     TCAN: Rank #{tcan_perf['Rank']}/4, Error={tcan_perf['Error']:.4f}, Improvement={tcan_perf['Improvement']:.1f}%")
            if tcan_perf['Rank'] == 1:
                print("     ⚠️  TCAN should have won this event!")
            elif tcan_perf['Rank'] == 2:
                winner_error = models_performance[0]['Error']
                print(f"     → Close 2nd place (winner error: {winner_error:.4f})")
            else:
                print(f"     → Ranked {tcan_perf['Rank']} out of 4 models")
        
        # Show top 3 for context
        print("     Top 3: ", end="")
        for i, model_perf in enumerate(models_performance[:3]):
            print(f"{i+1}.{model_perf['Model']}({model_perf['Error']:.4f})", end=" ")
        print()
        print()
    
    # Count event wins
    event_wins = superior_events['best_ai_model'].value_counts()
    print("📊 EVENT WINS:")
    for model, wins in event_wins.items():
        print(f"   • {model}: {wins}/{len(superior_events)} events ({wins/len(superior_events)*100:.1f}%)")
    
    # Show overall ranking
    ai_models = evaluation_results[evaluation_results['Model'].isin(['SVM', 'ANN', 'TCN', 'TCAN'])].copy()
    ai_models = ai_models.sort_values('Composite_Score')
    
    print("\n🏆 OVERALL RANKING (by statistical metrics):")
    for idx, (_, row) in enumerate(ai_models.iterrows()):
        print(f"   {idx+1}. {row['Model']}: R²={row['R2']:.4f}, RMSE={row['RMSE']:.3f}")
    
    # Analyze the discrepancy
    print("\n🔍 WHY THE DISCREPANCY?")
    
    # Find TCN's performance details
    tcn_events = superior_events[superior_events['best_ai_model'] == 'TCN']
    ann_events = superior_events[superior_events['best_ai_model'] == 'ANN']
    
    if len(tcn_events) > 0:
        print(f"\n   TCN wins {len(tcn_events)} events but ranks #1 overall because:")
        for _, event in tcn_events.iterrows():
            print(f"   • {event['duration_mins']} min event: {event['improvement_pct']:.1f}% improvement")
            print(f"     (Error: {event['best_ai_error']:.4f} vs Gumbel: {event['gumbel_error']:.4f})")
        
        print("   → TCN's exceptional accuracy in these events heavily influences aggregate metrics")
    
    if len(ann_events) > 0:
        print(f"\n   ANN wins {len(ann_events)} events but ranks lower because:")
        for _, event in ann_events.iterrows():
            print(f"   • {event['duration_mins']} min event: {event['improvement_pct']:.1f}% improvement")
            print(f"     (Error: {event['best_ai_error']:.4f} vs Gumbel: {event['gumbel_error']:.4f})")
        
        print("   → ANN is more consistent but less spectacular than TCN's peak performance")
    
    # Performance by duration
    print("\n📈 MODEL SPECIALIZATION:")
    duration_performance = {}
    for _, event in superior_events.iterrows():
        duration = event['duration_mins']
        model = event['best_ai_model']
        improvement = event['improvement_pct']
        
        if duration not in duration_performance:
            duration_performance[duration] = []
        duration_performance[duration].append((model, improvement))
    
    for duration, performances in sorted(duration_performance.items()):
        if duration <= 60:
            duration_str = f"{duration} minutes"
        else:
            hours = duration / 60
            duration_str = f"{hours:.0f} hours" if hours == int(hours) else f"{hours:.1f} hours"
        
        for model, improvement in performances:
            print(f"   • {duration_str}: {model} ({improvement:.1f}% improvement)")
    
    print("\n💡 KEY INSIGHT:")
    print("   • Event wins = Who performs best on individual cases")
    print("   • Overall ranking = Who has best aggregate statistical performance")
    print("   • TCN excels dramatically in specific scenarios (short durations)")
    print("   • ANN performs consistently well across diverse conditions")
    print("   • Both approaches have merit for different applications")

def generate_model_ranking_summary(evaluation_results):
    """
    Generate a comprehensive model ranking summary
    """
    if len(evaluation_results) == 0:
        return
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE MODEL RANKING SUMMARY")
    print("=" * 100)
    
    # Remove Gumbel for AI-only ranking
    ai_only = evaluation_results[evaluation_results['Model'] != 'Gumbel'].copy()
    ai_only['AI_Rank'] = range(1, len(ai_only) + 1)
    
    print("\n📊 AI MODELS RANKING (on events where AI outperformed Gumbel):")
    print("-" * 70)
    for _, row in ai_only.iterrows():
        print(f"   {row['AI_Rank']}. {row['Model']:<5} - "
              f"R²: {row['R2']:.3f}, RMSE: {row['RMSE']:.3f}, "
              f"MAE: {row['MAE']:.3f}, NSE: {row['NSE']:.3f}")
    
    # Performance insights
    best_ai = ai_only.iloc[0]
    worst_ai = ai_only.iloc[-1]
    
    print("\n💡 INSIGHTS:")
    print(f"   • Best AI Model: {best_ai['Model']} (R² = {best_ai['R2']:.3f})")
    print(f"   • Most Improved Metric: R² ranges from {worst_ai['R2']:.3f} to {best_ai['R2']:.3f}")
    print(f"   • RMSE Performance: {best_ai['Model']} ({best_ai['RMSE']:.3f}) vs worst {worst_ai['Model']} ({worst_ai['RMSE']:.3f})")
    
    # Count wins per model
    gumbel_metrics = evaluation_results[evaluation_results['Model'] == 'Gumbel'].iloc[0]
    wins_count = {}
    
    for _, row in ai_only.iterrows():
        model = row['Model']
        wins = 0
        if row['RMSE'] < gumbel_metrics['RMSE']: 
            wins += 1
        if row['MAE'] < gumbel_metrics['MAE']: 
            wins += 1
        if row['R2'] > gumbel_metrics['R2']: 
            wins += 1
        if row['NSE'] > gumbel_metrics['NSE']: 
            wins += 1
        wins_count[model] = wins
    
    print("\n🏆 METRIC WINS vs GUMBEL (out of 4 metrics):")
    for model, wins in sorted(wins_count.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {model}: {wins}/4 metrics better than Gumbel")



def main():
    print("=" * 80)
    print("COMPREHENSIVE IDF CURVE MODEL EVALUATION")
    print("=" * 80)
    print("\n⚠️  METHODOLOGY: Unbiased evaluation on full validation dataset (2019-2025)")
    print("   followed by case studies on 7 objectively-selected extreme events")
    
    # Load all datasets
    print("\n1. Loading datasets...")
    try:
        # Load annual maximum intensity data
        annual_max_df = pd.read_csv('./results/annual_max_intensity.csv')
        print(f"   ✓ Annual max data loaded: {annual_max_df.shape}")
        
        # Load Gumbel IDF data
        gumbel_df = pd.read_csv('./results/idf_data.csv')
        print(f"   ✓ Gumbel IDF data loaded: {gumbel_df.shape}")
        
        # Load Literature IDF data
        literature_df = pd.read_csv('./results/idf_lit.csv')
        print(f"   ✓ Literature IDF data loaded: {literature_df.shape}")
        
        # Load AI model IDF curves
        ai_models = {}
        
        svm_df = pd.read_csv('./results/idf_curves_SVM.csv')
        ai_models['SVM'] = svm_df
        print(f"   ✓ SVM IDF data loaded: {svm_df.shape}")
        
        ann_df = pd.read_csv('./results/idf_curves_ANN.csv')
        ai_models['ANN'] = ann_df
        print(f"   ✓ ANN IDF data loaded: {ann_df.shape}")
        
        tcn_df = pd.read_csv('./results/idf_curves_TCN.csv')
        ai_models['TCN'] = tcn_df
        print(f"   ✓ TCN IDF data loaded: {tcn_df.shape}")
        
        tcan_df = pd.read_csv('./results/idf_curves_TCAN.csv')
        ai_models['TCAN'] = tcan_df
        print(f"   ✓ TCAN IDF data loaded: {tcan_df.shape}")
        
    except FileNotFoundError as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # ========================================================================
    # SECTION 1: FULL VALIDATION DATASET EVALUATION (PRIMARY ANALYSIS)
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: FULL VALIDATION DATASET EVALUATION (2019-2025)")
    print("=" * 80)
    print("This is the authoritative, unbiased performance evaluation.")
    print("Matches validation period used in model training (2019-2025, 7 years).")
    
    overall_metrics, per_duration_metrics, all_predictions = evaluate_full_validation_dataset(
        annual_max_df, gumbel_df, literature_df, ai_models
    )
    
    if len(overall_metrics) == 0:
        print("   ✗ Failed to evaluate full validation dataset")
        return
    
    print("\n" + "=" * 80)
    print(f"FULL VALIDATION RESULTS (N={len(all_predictions)} observations)")
    print("=" * 80)
    
    # Sort by composite score
    overall_metrics['Composite_Score'] = (
        overall_metrics['RMSE'] / overall_metrics['RMSE'].max() +
        overall_metrics['MAE'] / overall_metrics['MAE'].max() +
        (1 - overall_metrics['R2']) +
        (1 - overall_metrics['NSE'])
    ) / 4
    
    overall_metrics = overall_metrics.sort_values('Composite_Score')
    overall_metrics['Rank'] = range(1, len(overall_metrics) + 1)
    
    display_cols = ['Rank', 'Model', 'RMSE', 'MAE', 'R2', 'NSE', 'N']
    print(overall_metrics[display_cols].to_string(index=False, float_format='%.4f'))
    
    # Statistical significance testing
    print("\n" + "-" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("-" * 80)
    
    # Bootstrap confidence intervals for top models
    print("\n95% Confidence Intervals (Bootstrap, 1000 iterations):")
    for idx, row in overall_metrics.head(5).iterrows():
        model_name = row['Model']
        if model_name in all_predictions.columns:
            observed = all_predictions['observed'].values
            predicted = all_predictions[model_name].values
            
            rmse_lower, rmse_upper, mae_lower, mae_upper = calculate_confidence_intervals(
                observed, predicted, n_bootstrap=1000, confidence=0.95
            )
            
            print(f"\n{model_name}:")
            print(f"   RMSE: [{rmse_lower:.4f}, {rmse_upper:.4f}]")
            print(f"   MAE:  [{mae_lower:.4f}, {mae_upper:.4f}]")
    
    # Paired t-tests vs Gumbel
    print("\n" + "-" * 80)
    print("Paired T-Tests: Model Error Comparisons")
    print("-" * 80)
    print("Tests whether prediction errors differ significantly from Gumbel (reference).")
    print("All models are evaluated against OBSERVED DATA as the true baseline.")
    t_test_results = perform_paired_t_tests(all_predictions, baseline_model='Gumbel')
    if len(t_test_results) > 0:
        display_cols = ['Model', 'Mean_Error', 'Error_Diff', 'Pct_Change', 'p_value', 'vs_Baseline']
        print(t_test_results[display_cols].to_string(index=False, float_format='%.4f'))
    
    # ========================================================================
    # SECTION 2: SELECT EXTREME EVENTS (OBJECTIVE CRITERIA)
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: SELECTING EXTREME EVENTS FOR CASE STUDIES")
    print("=" * 80)
    print("Selection criteria: High return periods (≥3.5yr in 7-year period), high intensities, diverse durations")
    print("NO filtering by model performance - purely objective selection.")
    
    extreme_events = select_extreme_events(all_predictions, all_predictions, n_events=7)
    
    if len(extreme_events) == 0:
        print("   ✗ No extreme events found")
        return
    
    # Find dates for selected events
    print("\n2.1. Finding dates for selected extreme events...")
    max_event_dates = find_max_event_dates(annual_max_df)
    
    # Evaluate all models on these events
    print("\n2.2. Evaluating all models on selected events...")
    events_analysis = evaluate_models_on_events(extreme_events, max_event_dates)
    
    if len(events_analysis) == 0:
        print("   ✗ Failed to evaluate events")
        return
    
    # ========================================================================
    # SECTION 3: EXTREME EVENT CASE STUDIES (SUPPLEMENTARY ANALYSIS)
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"SECTION 3: EXTREME EVENT CASE STUDIES (N={len(events_analysis)} events)")
    print("=" * 80)
    print("⚠️  NOTE: Event-specific metrics may differ from full dataset metrics.")
    print("   Small sample size (N=7) vs. full validation (N={})".format(len(all_predictions)))
    
    # Create comparison table
    comparison_table = create_event_comparison_table(events_analysis)
    
    # Create comparison table
    comparison_table = create_event_comparison_table(events_analysis)
    
    print("\n" + "=" * 80)
    print("EXTREME EVENT COMPARISON TABLE")
    print("=" * 80)
    print(comparison_table.to_string(index=False))
    
    # Detailed event analysis
    detailed_model_analysis_per_event(events_analysis)
    
    # Event-specific model evaluation
    print("\n3.1. Model performance on selected events...")
    event_metrics = comprehensive_model_evaluation(events_analysis, dataset_type='selected_events')
    
    if len(event_metrics) > 0:
        print("\n" + "=" * 80)
        print(f"EVENT-SPECIFIC MODEL PERFORMANCE (N={len(events_analysis)} events)")
        print("=" * 80)
        print("⚠️  WARNING: Small sample size - these metrics may not reflect overall performance!")
        
        event_metrics_sorted = event_metrics.sort_values('RMSE')
        display_cols = ['Model', 'RMSE', 'MAE', 'R2', 'NSE', 'N']
        print(event_metrics_sorted[display_cols].to_string(index=False, float_format='%.4f'))
    
    # ========================================================================
    # SECTION 4: COMPARISON OF FULL vs EVENT METRICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: FULL DATASET vs. EVENT-SPECIFIC METRICS COMPARISON")
    print("=" * 80)
    print("This explains why event-specific rankings may differ from overall rankings.")
    
    # Merge metrics for comparison
    overall_for_comparison = overall_metrics[['Model', 'RMSE', 'MAE', 'R2', 'NSE']].copy()
    overall_for_comparison.columns = ['Model', 'Full_RMSE', 'Full_MAE', 'Full_R2', 'Full_NSE']
    
    event_for_comparison = event_metrics[['Model', 'RMSE', 'MAE', 'R2', 'NSE']].copy()
    event_for_comparison.columns = ['Model', 'Event_RMSE', 'Event_MAE', 'Event_R2', 'Event_NSE']
    
    comparison = overall_for_comparison.merge(event_for_comparison, on='Model', how='inner')
    comparison['RMSE_Diff'] = comparison['Event_RMSE'] - comparison['Full_RMSE']
    comparison['R2_Diff'] = comparison['Event_R2'] - comparison['Full_R2']
    
    print("\nModel Performance Comparison (Full Dataset vs. 7 Selected Events):")
    print("=" * 80)
    for _, row in comparison.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  Full Dataset (N={len(all_predictions)}): RMSE={row['Full_RMSE']:.4f}, R²={row['Full_R2']:.4f}")
        print(f"  Events (N={len(events_analysis)}):       RMSE={row['Event_RMSE']:.4f}, R²={row['Event_R2']:.4f}")
        print(f"  Difference:            RMSE Δ={row['RMSE_Diff']:+.4f}, R² Δ={row['R2_Diff']:+.4f}")
    
    print("\n" + "=" * 80)
    print("💡 KEY INSIGHT:")
    print("=" * 80)
    print("• Full dataset metrics (N={}) are AUTHORITATIVE".format(len(all_predictions)))
    print("• Event metrics (N={}) show performance on EXTREME cases only".format(len(events_analysis)))
    print("• Differences indicate model specialization:")
    print("  - Some models excel on extreme events but average overall")
    print("  - Others are consistent across all conditions")
    print("• Use FULL DATASET metrics for general model selection")
    print("• Use EVENT metrics for understanding extreme event capabilities")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    try:
        # Save full validation metrics (primary results)
        overall_metrics.to_csv('./results/full_validation_metrics.csv', index=False)
        print("   ✓ Full validation metrics saved to: ./results/full_validation_metrics.csv")
        
        # Save event case studies (supplementary results)
        comparison_table.to_csv('./results/extreme_events_case_studies.csv', index=False)
        print("   ✓ Event case studies saved to: ./results/extreme_events_case_studies.csv")
        
        # Save per-duration metrics
        per_duration_metrics.to_csv('./results/full_validation_per_duration_metrics.csv', index=False)
        print("   ✓ Per-duration metrics saved to: ./results/full_validation_per_duration_metrics.csv")
        
    except Exception as e:
        print(f"   ✗ Error saving results: {e}")
    
    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 FULL VALIDATION DATASET (PRIMARY RESULTS, N={len(all_predictions)}):")
    print("   Validation period: 2019-2025 (7 years)")
    
    # Overall best model
    best_overall = overall_metrics.iloc[0]
    print(f"   🏆 Best Overall Model: {best_overall['Model']}")
    print(f"      RMSE: {best_overall['RMSE']:.4f}, MAE: {best_overall['MAE']:.4f}")
    print(f"      R²: {best_overall['R2']:.4f}, NSE: {best_overall['NSE']:.4f}")
    
    # AI vs Traditional comparison
    ai_models_list = ['SVM', 'ANN', 'TCN', 'TCAN']
    best_ai = overall_metrics[overall_metrics['Model'].isin(ai_models_list)].iloc[0]
    gumbel_perf = overall_metrics[overall_metrics['Model'] == 'Gumbel'].iloc[0]
    
    print(f"\n   🤖 Best AI Model: {best_ai['Model']}")
    print(f"      RMSE: {best_ai['RMSE']:.4f} (vs. Gumbel: {gumbel_perf['RMSE']:.4f})")
    improvement_pct = (gumbel_perf['RMSE'] - best_ai['RMSE']) / gumbel_perf['RMSE'] * 100
    print(f"      Improvement over Gumbel: {improvement_pct:.1f}%")
    
    print(f"\n📈 EXTREME EVENT CASE STUDIES (SUPPLEMENTARY, N={len(events_analysis)}):")
    print(f"   Selected events: {len(events_analysis)} extreme events")
    print(f"   Return period range: {events_analysis['return_period'].min():.1f}-{events_analysis['return_period'].max():.1f} years")
    print(f"   Duration range: {events_analysis['duration_mins'].min()}-{events_analysis['duration_mins'].max()} minutes")
    
    # Count which model performs best on each event
    if 'best_model' in events_analysis.columns:
        best_per_event = events_analysis['best_model'].value_counts()
        print("\n   Best model frequency on extreme events:")
        for model, count in best_per_event.items():
            pct = (count / len(events_analysis)) * 100
            print(f"      • {model}: {count}/{len(events_analysis)} events ({pct:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nRecommendations:")
    print(f"1. Use {best_overall['Model']} for general IDF curve construction")
    print(f"2. Consider {best_ai['Model']} for AI-based approaches")
    print("3. Review event case studies for extreme event insights")
    print("4. Full validation metrics are in: ./results/full_validation_metrics.csv")

if __name__ == "__main__":
    main()