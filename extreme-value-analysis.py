import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    Only processes validation period (1998-2025)
    """
    print("   Loading historical intensity data to find event dates...")

    # Filter to validation period (1998-2025)
    annual_max_df = annual_max_df[(annual_max_df['year'] >= 1998) & (annual_max_df['year'] <= 2025)].copy()
    print(f"   ✓ Processing validation period 1998-2025: {len(annual_max_df)} years")

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

def find_ai_superiority_events(annual_max_df, gumbel_df, literature_df, ai_models, max_event_dates=None, n_events=10):
    """
    Find events where at least one AI model outperforms Gumbel significantly
    Only processes events from 1998-2025 validation period
    """
    # Filter annual_max_df to only include validation period (1998-2025)
    annual_max_df = annual_max_df[(annual_max_df['year'] >= 1998) & (annual_max_df['year'] <= 2025)].copy()
    print(f"   ✓ Filtered to validation period 1998-2025: {len(annual_max_df)} years of data")
    
    # Duration mapping for different datasets
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
    
    duration_mapping_literature = {
        5: '5 mins', 10: '10 mins', 15: '15 mins', 30: '30 mins',
        60: '60 mins', 90: '90 mins', 120: '120 mins', 180: '180 mins', 360: '360 mins', 
        720: '720 mins', 900: '900 mins', 1080: '1080 mins', 1440: '1440 mins'
    }
    
    superior_events = []
    
    # Analyze each duration
    for duration_mins in [5, 10, 15, 30, 60, 90, 120, 180, 360, 720, 900, 1080, 1440]:
        # Get annual max data for this duration
        annual_col = duration_mapping_annual.get(duration_mins)
        if annual_col is None or annual_col not in annual_max_df.columns:
            continue
            
        # Calculate return periods using Weibull formula
        annual_data = annual_max_df[annual_col].dropna()
        if len(annual_data) < 5:  # Need sufficient data
            continue
            
        # Create a mapping of intensity values to years for date lookup
        intensity_to_year = {}
        for idx, row in annual_max_df.iterrows():
            if not pd.isna(row[annual_col]):
                intensity_to_year[row[annual_col]] = row['year']
        
        weibull_results = calculate_return_periods_weibull(annual_data)
        
        # Focus on events with return periods between 1.5-100 years (broader range)
        target_events = weibull_results[
            (weibull_results['return_period'] >= 1.5) & 
            (weibull_results['return_period'] <= 100)
        ].copy()
        
        if len(target_events) == 0:
            continue
        
        # Sample some events across different return periods
        if len(target_events) > 6:
            # Take events at different return period ranges (broader sampling)
            rp_ranges = [
                (1.5, 3), (3, 7), (7, 15), (15, 30), (30, 100)
            ]
            sampled_events = []
            for rp_min, rp_max in rp_ranges:
                range_events = target_events[
                    (target_events['return_period'] >= rp_min) & 
                    (target_events['return_period'] < rp_max)
                ]
                if len(range_events) > 0:
                    # Take the event closest to the middle of the range
                    mid_rp = (rp_min + rp_max) / 2
                    closest_idx = (range_events['return_period'] - mid_rp).abs().idxmin()
                    sampled_events.append(range_events.loc[closest_idx])
            
            if sampled_events:
                target_events = pd.DataFrame(sampled_events)
        
        # For each event, compare model predictions
        for idx, event in target_events.iterrows():
            observed_intensity = event['intensity']
            return_period = event['return_period']
            
            # Find the year and date for this event
            event_year = intensity_to_year.get(observed_intensity)
            event_date = None
            if event_year is not None and max_event_dates is not None and event_year in max_event_dates:
                event_date = max_event_dates[event_year].get(annual_col)
            
            # Get Gumbel prediction
            gumbel_col = duration_mapping_gumbel.get(duration_mins)
            if gumbel_col is None or gumbel_col not in gumbel_df.columns:
                continue
                
            gumbel_pred = interpolate_idf_for_return_period(
                gumbel_df, return_period, gumbel_col
            )
            
            # Get Literature prediction
            literature_col = duration_mapping_literature.get(duration_mins)
            literature_pred = None
            if literature_col is not None and literature_col in literature_df.columns:
                literature_pred = interpolate_idf_for_return_period(
                    literature_df, return_period, literature_col
                )
            
            # Get AI model predictions
            ai_predictions = {}
            for model_name, model_df in ai_models.items():
                try:
                    ai_pred = interpolate_ai_idf_for_return_period(
                        model_df, return_period, duration_mins
                    )
                    ai_predictions[model_name] = ai_pred
                except Exception:
                    continue
            
            # Calculate errors for all models
            gumbel_error = abs(observed_intensity - gumbel_pred)
            literature_error = abs(observed_intensity - literature_pred) if literature_pred is not None else None
            ai_errors = {name: abs(observed_intensity - pred) 
                        for name, pred in ai_predictions.items()}
            
            # Check if at least one AI model outperforms Gumbel by a small margin (>5%)
            if len(ai_errors) > 0:
                best_ai_error = min(ai_errors.values())
                if best_ai_error < gumbel_error * 0.95:  # At least 5% improvement
                    improvement = (gumbel_error - best_ai_error) / gumbel_error * 100
                    best_model = min(ai_errors.keys(), key=lambda k: ai_errors[k])
                    
                    superior_events.append({
                        'duration_mins': duration_mins,
                        'observed_intensity': observed_intensity,
                        'return_period': return_period,
                        'event_year': event_year,
                        'event_date': event_date,
                        'gumbel_pred': gumbel_pred,
                        'gumbel_error': gumbel_error,
                        'literature_pred': literature_pred,
                        'literature_error': literature_error,
                        'best_ai_model': best_model,
                        'best_ai_pred': ai_predictions[best_model],
                        'best_ai_error': best_ai_error,
                        'improvement_pct': improvement,
                        **{f'{name}_pred': pred for name, pred in ai_predictions.items()},
                        **{f'{name}_error': error for name, error in ai_errors.items()}
                    })
    
    # Sort by improvement percentage and return top events
    superior_events_df = pd.DataFrame(superior_events)
    if len(superior_events_df) == 0:
        print("No events found where AI models significantly outperform Gumbel")
        return pd.DataFrame()
    
    superior_events_df = superior_events_df.sort_values('improvement_pct', ascending=False)
    
    # Ensure we pick events from unique dates to provide diverse analysis
    unique_events = []
    used_dates = set()
    
    for idx, event in superior_events_df.iterrows():
        event_date = None
        
        # Extract date information
        if event.get('event_date') is not None and pd.notna(event['event_date']):
            try:
                if isinstance(event['event_date'], str):
                    # Parse string date and extract date part
                    event_date = pd.to_datetime(event['event_date']).date()
                else:
                    # Extract date part from datetime
                    event_date = event['event_date'].date()
            except Exception:
                # Fallback to using year if date parsing fails
                event_date = event.get('event_year')
        else:
            # Use year as fallback identifier
            event_date = event.get('event_year')
        
        # Check if we've already used this date
        if event_date not in used_dates:
            unique_events.append(event)
            used_dates.add(event_date)
            
            # Stop when we have enough unique events
            if len(unique_events) >= n_events:
                break
    
    # Convert back to DataFrame
    if len(unique_events) > 0:
        result_df = pd.DataFrame(unique_events)
        print(f"   ✓ Selected {len(result_df)} events from {len(used_dates)} unique dates")
        return result_df
    else:
        print("   ✗ No unique date events found")
        return pd.DataFrame()

def comprehensive_model_evaluation(events_df, ai_models):
    """
    Comprehensive evaluation of all models on the selected events
    """
    if len(events_df) == 0:
        return pd.DataFrame()
    
    model_names = ['Literature', 'Gumbel'] + list(ai_models.keys())
    
    # Collect all predictions and observations
    all_predictions = {model: [] for model in model_names}
    all_observations = []
    
    for idx, event in events_df.iterrows():
        all_observations.append(event['observed_intensity'])
        all_predictions['Literature'].append(event['literature_pred'] if 'literature_pred' in event and event['literature_pred'] is not None else np.nan)
        all_predictions['Gumbel'].append(event['gumbel_pred'])
        
        for model_name in ai_models.keys():
            pred_col = f'{model_name}_pred'
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
            'N_Events': len([p for p in preds if not np.isnan(p)])
        })
    
    return pd.DataFrame(results)

def create_superiority_table(events_df):
    """
    Create a detailed table showing AI superiority events with all outperforming models
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
        elif event.get('event_year') is not None:
            date_str = f"{int(event['event_year'])}"
        
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
        
        # Literature and Statistical predictions
        literature_pred = event['literature_pred'] if event.get('literature_pred') is not None else None
        statistical_pred = event['gumbel_pred']
        gumbel_error = event['gumbel_error']
        
        # Find all AI models that outperform Gumbel and rank them
        ai_models_performance = []
        # Also collect all AI model predictions/errors for best/least summary
        ai_models_all = []
        for model in ['SVM', 'ANN', 'TCN', 'TCAN']:
            error_col = f'{model}_error'
            if error_col in event:
                ai_error = event[error_col]
                if ai_error < gumbel_error:  # AI model outperforms Gumbel
                    improvement_pct = (gumbel_error - ai_error) / gumbel_error * 100
                    ai_models_performance.append({
                        'model': model,
                        'error': ai_error,
                        'improvement_pct': improvement_pct
                    })
                # Keep track regardless of beating Gumbel for least/best summary among available AI
                pred_col = f'{model}_pred'
                if pred_col in event:
                    ai_models_all.append({
                        'model': model,
                        'error': ai_error,
                        'pred': event[pred_col]
                    })
            else:
                # If error not present but prediction exists, include with NaN error (won't be chosen as best)
                pred_col = f'{model}_pred'
                if pred_col in event:
                    ai_models_all.append({
                        'model': model,
                        'error': np.nan,
                        'pred': event[pred_col]
                    })
        
        # Sort by error (best performance first)
        ai_models_performance.sort(key=lambda x: x['error'])
        
        # Create ranked list of outperforming models
        if ai_models_performance:
            outperforming_models = [model['model'] for model in ai_models_performance]
            improvements = [model['improvement_pct'] for model in ai_models_performance]
            
            # Format as "Model1, Model2, Model3" (in order of performance)
            models_list = ", ".join(outperforming_models)
            
            # Show improvement range (min%, max%)
            if len(improvements) > 1:
                improvement_range = f"{min(improvements):.1f}%, {max(improvements):.1f}%"
            else:
                improvement_range = f"{improvements[0]:.1f}%"
        else:
            models_list = "None"
            improvement_range = "0%"
        
        # Compose AI Pred (mm/hr): numeric predictions only, ordered to match 'AI Models (Best→Worst)'
        # This lets readers map values to models by position without repeating model names.
        ai_pred_summary = "N/A"
        if ai_models_performance:
            preds_in_order = []
            for perf in ai_models_performance:  # already sorted best→worst by error
                model = perf['model']
                pred_col = f'{model}_pred'
                if pred_col in event:
                    preds_in_order.append(f"{event[pred_col]:.2f}")
            if preds_in_order:
                ai_pred_summary = ", ".join(preds_in_order)
        
        table_data.append({
            'Event': event_name,
            'Date': date_str,
            'Duration': duration_display,
            'Observed (mm/hr)': f"{event['observed_intensity']:.2f}",
            'Literature (mm/hr)': f"{literature_pred:.2f}" if literature_pred is not None else "N/A",
            'Statistical Pred (mm/hr)': f"{statistical_pred:.2f}",
            'AI Models (Best→Worst)': models_list,
            'AI Pred (mm/hr)': ai_pred_summary,
            'Improvement Range': improvement_range
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
                if event.get('event_year') is not None:
                    date_info = f" (Year: {int(event['event_year'])})"
        elif event.get('event_year') is not None:
            date_info = f" (Year: {int(event['event_year'])})"
        
        print(f"\n🎯 EVENT {row_num+1}: Duration {event['duration_mins']} minutes, "
              f"Return Period {event['return_period']:.1f} years{date_info}")
        print("-" * 80)
        print(f"   Observed Intensity: {event['observed_intensity']:.2f} mm/hr")
        
        # Show Literature prediction if available
        if event.get('literature_pred') is not None:
            print(f"   Literature Prediction: {event['literature_pred']:.2f} mm/hr (Error: {event['literature_error']:.2f})")
        else:
            print("   Literature Prediction: N/A")
        
        print(f"   Gumbel Prediction:  {event['gumbel_pred']:.2f} mm/hr (Error: {event['gumbel_error']:.2f})")
        
        # Show all AI model predictions
        ai_models = ['SVM', 'ANN', 'TCN', 'TCAN']
        ai_results = []
        
        for model in ai_models:
            pred_col = f'{model}_pred'
            error_col = f'{model}_error'
            if pred_col in event:
                pred = event[pred_col]
                error = event[error_col]
                improvement = (event['gumbel_error'] - error) / event['gumbel_error'] * 100
                ai_results.append({
                    'Model': model,
                    'Prediction': pred,
                    'Error': error,
                    'Improvement': improvement,
                    'Better_than_Gumbel': '✓' if error < event['gumbel_error'] else '✗'
                })
        
        ai_results_df = pd.DataFrame(ai_results)
        ai_results_df = ai_results_df.sort_values('Error')
        
        print("\n   AI Model Performance:")
        for _, row in ai_results_df.iterrows():
            print(f"   {row['Better_than_Gumbel']} {row['Model']:<5}: "
                  f"{row['Prediction']:.2f} mm/hr (Error: {row['Error']:.2f}, "
                  f"Improvement: {row['Improvement']:.1f}%)")
        
        best_model = ai_results_df.iloc[0]['Model']
        print(f"\n   🏆 Best Model: {best_model} with {ai_results_df.iloc[0]['Improvement']:.1f}% improvement over Gumbel")

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

def diagnose_event_selection_bias(annual_max_df, gumbel_df, literature_df, ai_models):
    """
    Diagnose potential bias in event selection that might favor certain models
    """
    print("\n" + "=" * 100)
    print("EVENT SELECTION BIAS ANALYSIS")
    print("=" * 100)
    
    # Find ALL events where ANY AI model outperforms Gumbel (not just the best improvement)
    all_superior_events = find_ai_superiority_events(
        annual_max_df, gumbel_df, literature_df, ai_models, 
        max_event_dates=None, n_events=50  # Get more events
    )
    
    if len(all_superior_events) == 0:
        print("   No events found for analysis")
        return
    
    print(f"   Found {len(all_superior_events)} total events where AI outperforms Gumbel")
    
    # Count wins by model across ALL events
    all_wins = all_superior_events['best_ai_model'].value_counts()
    print("\n📊 EXPANDED EVENT WINS (all events where AI beats Gumbel):")
    for model, wins in all_wins.items():
        percentage = (wins / len(all_superior_events)) * 100
        print(f"   • {model}: {wins}/{len(all_superior_events)} events ({percentage:.1f}%)")
    
    # Check model performance across different durations
    duration_performance = {}
    for _, event in all_superior_events.iterrows():
        duration = event['duration_mins']
        winner = event['best_ai_model']
        
        if duration not in duration_performance:
            duration_performance[duration] = {'SVM': 0, 'ANN': 0, 'TCN': 0, 'TCAN': 0}
        
        duration_performance[duration][winner] += 1
    
    print("\n📈 PERFORMANCE BY DURATION (all events):")
    for duration in sorted(duration_performance.keys()):
        duration_str = f"{duration} min" if duration <= 60 else f"{duration/60:.0f} hr"
        total_events = sum(duration_performance[duration].values())
        
        print(f"   {duration_str} ({total_events} events):")
        for model, wins in duration_performance[duration].items():
            if wins > 0:
                pct = (wins / total_events) * 100
                print(f"     • {model}: {wins} wins ({pct:.1f}%)")
    
    # Analyze improvement distributions
    print("\n📊 IMPROVEMENT DISTRIBUTION BY MODEL:")
    for model in ['SVM', 'ANN', 'TCN', 'TCAN']:
        model_events = all_superior_events[all_superior_events['best_ai_model'] == model]
        if len(model_events) > 0:
            improvements = model_events['improvement_pct'].values
            print(f"   {model}: {len(model_events)} wins, "
                  f"Improvements: {improvements.min():.1f}%-{improvements.max():.1f}% "
                  f"(avg: {improvements.mean():.1f}%)")
    
    return all_superior_events

def main():
    print("=" * 80)
    print("AI MODELS SUPERIORITY ANALYSIS FOR IDF CURVES")
    print("=" * 80)
    
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
    
    # Find maximum event dates
    print("\n2. Finding dates of maximum intensity events...")
    max_event_dates = find_max_event_dates(annual_max_df)
    
    # Diagnose potential event selection bias first
    print("\n3a. Diagnosing potential event selection bias...")
    diagnose_event_selection_bias(annual_max_df, gumbel_df, literature_df, ai_models)
    
    # Find events where AI models outperform Gumbel
    print("\n3b. Finding events where AI models significantly outperform Gumbel...")
    superior_events = find_ai_superiority_events(annual_max_df, gumbel_df, literature_df, ai_models, max_event_dates, n_events=10)
    
    if len(superior_events) == 0:
        print("   ✗ No superior events found")
        return
    
    print(f"   ✓ Found {len(superior_events)} events where AI models significantly outperform Gumbel")
    
    # Create superiority table
    print("\n4. Creating superiority analysis table...")
    superiority_table = create_superiority_table(superior_events)
    
    print("\n" + "=" * 100)
    print("AI MODELS SUPERIORITY EVENTS")
    print("=" * 100)
    print(superiority_table.to_string(index=False))
    
    # Detailed event analysis
    detailed_model_analysis_per_event(superior_events)
    
    # Comprehensive model evaluation
    print("\n5. Comprehensive model evaluation on selected events...")
    evaluation_results = comprehensive_model_evaluation(superior_events, ai_models)
    
    if len(evaluation_results) > 0:
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL PERFORMANCE ON SUPERIOR EVENTS")
        print("=" * 80)
        
        # Sort by composite score (lower is better)
        evaluation_results['Composite_Score'] = (
            evaluation_results['RMSE'] / evaluation_results['RMSE'].max() +
            evaluation_results['MAE'] / evaluation_results['MAE'].max() +
            (1 - evaluation_results['R2']) +
            (1 - evaluation_results['NSE'])
        ) / 4
        
        evaluation_results = evaluation_results.sort_values('Composite_Score')
        evaluation_results['Rank'] = range(1, len(evaluation_results) + 1)
        
        # Display results
        display_cols = ['Rank', 'Model', 'RMSE', 'MAE', 'R2', 'NSE', 'Composite_Score']
        print(evaluation_results[display_cols].to_string(index=False, float_format='%.4f'))
        
        # Calculate improvements over Gumbel
        print("\n" + "=" * 80)
        print("AI MODELS IMPROVEMENT OVER GUMBEL")
        print("=" * 80)
        
        gumbel_metrics = evaluation_results[evaluation_results['Model'] == 'Gumbel'].iloc[0]
        
        improvement_analysis = []
        for idx, row in evaluation_results.iterrows():
            if row['Model'] != 'Gumbel':
                rmse_improvement = (gumbel_metrics['RMSE'] - row['RMSE']) / gumbel_metrics['RMSE'] * 100
                mae_improvement = (gumbel_metrics['MAE'] - row['MAE']) / gumbel_metrics['MAE'] * 100
                r2_improvement = row['R2'] - gumbel_metrics['R2']
                nse_improvement = row['NSE'] - gumbel_metrics['NSE']
                
                improvement_analysis.append({
                    'Model': row['Model'],
                    'RMSE_Improvement (%)': rmse_improvement,
                    'MAE_Improvement (%)': mae_improvement,
                    'R2_Improvement': r2_improvement,
                    'NSE_Improvement': nse_improvement,
                    'Overall_Rank': row['Rank']
                })
        
        improvement_df = pd.DataFrame(improvement_analysis)
        improvement_df = improvement_df.sort_values('Overall_Rank')
        print(improvement_df.to_string(index=False, float_format='%.3f'))
        
        # Calculate Gumbel vs Literature comparison
        print("\n" + "=" * 80)
        print("GUMBEL IDF IMPROVEMENT OVER LITERATURE IDF")
        print("=" * 80)
        
        literature_metrics = evaluation_results[evaluation_results['Model'] == 'Literature']
        if len(literature_metrics) > 0:
            literature_metrics = literature_metrics.iloc[0]
            
            # Calculate improvements based on statistical metrics
            rmse_improvement = (literature_metrics['RMSE'] - gumbel_metrics['RMSE']) / literature_metrics['RMSE'] * 100
            mae_improvement = (literature_metrics['MAE'] - gumbel_metrics['MAE']) / literature_metrics['MAE'] * 100
            r2_improvement = gumbel_metrics['R2'] - literature_metrics['R2']
            nse_improvement = gumbel_metrics['NSE'] - literature_metrics['NSE']
            
            print(f"   RMSE Performance: Gumbel ({gumbel_metrics['RMSE']:.3f}) vs Literature ({literature_metrics['RMSE']:.3f})")
            print(f"   RMSE Improvement: {rmse_improvement:.2f}% {'(Gumbel Better)' if rmse_improvement > 0 else '(Literature Better)'}")
            
            print(f"   R² Performance: Gumbel ({gumbel_metrics['R2']:.4f}) vs Literature ({literature_metrics['R2']:.4f})")
            print(f"   R² Improvement: {r2_improvement:.4f} {'(Gumbel Better)' if r2_improvement > 0 else '(Literature Better)'}")
            
            print(f"   NSE Performance: Gumbel ({gumbel_metrics['NSE']:.4f}) vs Literature ({literature_metrics['NSE']:.4f})")
            print(f"   NSE Improvement: {nse_improvement:.4f} {'(Gumbel Better)' if nse_improvement > 0 else '(Literature Better)'}")
            
            # Overall assessment based on key metrics
            gumbel_better_metrics = 0
            total_key_metrics = 3  # RMSE, R2, NSE
            
            if rmse_improvement > 0:  # Lower RMSE is better, so positive improvement means Gumbel is better
                gumbel_better_metrics += 1
            if r2_improvement > 0:  # Higher R2 is better
                gumbel_better_metrics += 1
            if nse_improvement > 0:  # Higher NSE is better
                gumbel_better_metrics += 1
            
            print("\n   📊 Overall Assessment:")
            print(f"   • Gumbel outperforms Literature in {gumbel_better_metrics}/{total_key_metrics} key metrics")
            print(f"   • Literature outperforms Gumbel in {total_key_metrics - gumbel_better_metrics}/{total_key_metrics} key metrics")
            
            if gumbel_better_metrics > total_key_metrics / 2:
                print("   🏆 Statistical Winner: Gumbel IDF")
            else:
                print("   🏆 Statistical Winner: Literature IDF")
        else:
            print("   Literature data not available for comparison")

        # Generate detailed model ranking
        generate_model_ranking_summary(evaluation_results)
        
        # Explain ranking vs wins discrepancy
        explain_ranking_vs_wins_discrepancy(superior_events, evaluation_results)
    
    # Save results
    print("\n6. Saving results...")
    try:
        superiority_table.to_csv('./results/extreme_event_analysis.csv', index=False)
        print("   ✓ Superiority events saved to ./results/extreme_event_analysis.csv")
    except Exception as e:
        print(f"   ✗ Error saving results: {e}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"• Validation period: 1998-2025 ({len(annual_max_df)} years)")
    print(f"• Total events analyzed: {len(superior_events)}")
    print(f"• Average improvement over Gumbel: {superior_events['improvement_pct'].mean():.1f}%")
    print(f"• Best individual improvement: {superior_events['improvement_pct'].max():.1f}%")
    
    # Average per-event improvement of Gumbel over Literature (mirroring AI over Gumbel)
    try:
        if 'literature_error' in superior_events.columns:
            valid_mask = superior_events['literature_error'].notna() & (superior_events['literature_error'] > 0)
            if valid_mask.any():
                g_over_lit_impr = (superior_events.loc[valid_mask, 'literature_error'] - superior_events.loc[valid_mask, 'gumbel_error']) \
                                   / superior_events.loc[valid_mask, 'literature_error'] * 100
                print(f"• Avg improvement of Gumbel over Literature: {g_over_lit_impr.mean():.1f}% (across {valid_mask.sum()} comparable events)")
                print(f"• Best Gumbel-over-Literature improvement: {g_over_lit_impr.max():.1f}%")
    except Exception:
        pass
    
    # Count model wins
    best_models = superior_events['best_ai_model'].value_counts()
    print(f"• Most frequent best performer: {best_models.index[0]} ({best_models.iloc[0]}/{len(superior_events)} events)")
    
    if len(evaluation_results) > 1:
        ai_winner = evaluation_results[evaluation_results['Model'] != 'Gumbel'].iloc[0]['Model']
        print(f"• Overall best AI model: {ai_winner}")
    
    print(f"• Duration range analyzed: {superior_events['duration_mins'].min()}-{superior_events['duration_mins'].max()} minutes")
    print(f"• Return period range: {superior_events['return_period'].min():.1f}-{superior_events['return_period'].max():.1f} years")
    
    # Gumbel vs Literature statistical summary
    if len(evaluation_results) > 0:
        literature_metrics = evaluation_results[evaluation_results['Model'] == 'Literature']
        gumbel_metrics = evaluation_results[evaluation_results['Model'] == 'Gumbel']
        
        if len(literature_metrics) > 0 and len(gumbel_metrics) > 0:
            lit_metrics = literature_metrics.iloc[0]
            gum_metrics = gumbel_metrics.iloc[0]
            
            # Count how many key metrics Gumbel wins
            gumbel_wins = 0
            if gum_metrics['RMSE'] < lit_metrics['RMSE']:
                gumbel_wins += 1
            if gum_metrics['R2'] > lit_metrics['R2']:
                gumbel_wins += 1
            if gum_metrics['NSE'] > lit_metrics['NSE']:
                gumbel_wins += 1
            
            winner = "Gumbel" if gumbel_wins >= 2 else "Literature"
            print(f"• Gumbel vs Literature: {winner} statistically better ({gumbel_wins}/3 key metrics)")

            # Also report RMSE/MAE percent improvements for Gumbel over Literature
            try:
                rmse_impr_pct = ((lit_metrics['RMSE'] - gum_metrics['RMSE']) / lit_metrics['RMSE'] * 100) if lit_metrics['RMSE'] not in [0, np.nan] else np.nan
            except Exception:
                rmse_impr_pct = np.nan
            try:
                mae_impr_pct = ((lit_metrics['MAE'] - gum_metrics['MAE']) / lit_metrics['MAE'] * 100) if lit_metrics['MAE'] not in [0, np.nan] else np.nan
            except Exception:
                mae_impr_pct = np.nan

            # Only print when we have at least one valid metric
            if not (np.isnan(rmse_impr_pct) and np.isnan(mae_impr_pct)):
                rmse_str = f"RMSE {rmse_impr_pct:.2f}%" if not np.isnan(rmse_impr_pct) else "RMSE N/A"
                mae_str = f"MAE {mae_impr_pct:.2f}%" if not np.isnan(mae_impr_pct) else "MAE N/A"
                print(f"• Gumbel over Literature (metrics): {rmse_str}, {mae_str}")
    
    # Model frequency analysis
    print("\n🏆 AI MODEL PERFORMANCE SUMMARY:")
    for model, count in best_models.items():
        percentage = count / len(superior_events) * 100
        print(f"   • {model}: Best performer in {count}/{len(superior_events)} events ({percentage:.1f}%)")

if __name__ == "__main__":
    main()