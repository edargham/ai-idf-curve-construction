"""
Uncertainty Analysis for Statistical IDF Models

This module provides functions to quantify uncertainty in statistical distribution-based
IDF curve predictions, including prediction intervals and confidence bounds.
"""

import numpy as np
import pandas as pd
from scipy import stats
import os


def calculate_prediction_intervals(predictions, observations, confidence_level=0.95):
    """
    Calculate prediction intervals for statistical models.
    
    Parameters:
        predictions (array): Model predictions
        observations (array): Observed values
        confidence_level (float): Confidence level (default 0.95 for 95% CI)
        
    Returns:
        dict: Dictionary containing uncertainty metrics
    """
    predictions = np.array(predictions)
    observations = np.array(observations)
    
    # Calculate residuals
    residuals = observations - predictions
    
    # Standard deviation of residuals (aleatoric uncertainty proxy)
    residual_std = np.std(residuals)
    
    # Mean absolute deviation
    mad = np.mean(np.abs(residuals))
    
    # Calculate prediction interval width
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    prediction_interval_width = 2 * z_score * residual_std
    
    # Calculate coefficient of variation of residuals
    if np.mean(observations) != 0:
        cv_residuals = residual_std / np.mean(np.abs(observations))
    else:
        cv_residuals = np.nan
    
    # Percentage of observations within prediction interval
    lower_bound = predictions - z_score * residual_std
    upper_bound = predictions + z_score * residual_std
    coverage = np.mean((observations >= lower_bound) & (observations <= upper_bound))
    
    return {
        'residual_std': residual_std,
        'mean_absolute_deviation': mad,
        'prediction_interval_width': prediction_interval_width,
        'cv_residuals': cv_residuals,
        'prediction_interval_coverage': coverage,
        'confidence_level': confidence_level
    }


def calculate_distribution_uncertainty(params_dict, distribution_type, return_periods, validation_data):
    """
    Calculate uncertainty metrics for fitted statistical distributions.
    
    Parameters:
        params_dict (dict): Dictionary of distribution parameters for each duration
        distribution_type (str): Type of distribution ('gev', 'gumbel', etc.)
        return_periods (array): Return periods
        validation_data (DataFrame): Validation dataset
        
    Returns:
        dict: Uncertainty metrics including parameter uncertainty
    """
    # Parameter variability across durations
    if distribution_type in ['gev', 'weibull', 'lognormal']:
        # Three-parameter distributions
        shapes = [params_dict[dur][0] for dur in params_dict.keys()]
        locs = [params_dict[dur][1] for dur in params_dict.keys()]
        scales = [params_dict[dur][2] for dur in params_dict.keys()]
        
        param_std = {
            'shape_std': np.std(shapes),
            'location_std': np.std(locs),
            'scale_std': np.std(scales)
        }
    elif distribution_type == 'gumbel':
        # Two-parameter distribution
        locs = [params_dict[dur][0] for dur in params_dict.keys()]
        scales = [params_dict[dur][1] for dur in params_dict.keys()]
        
        param_std = {
            'location_std': np.std(locs),
            'scale_std': np.std(scales)
        }
    elif distribution_type == 'logpearson3':
        # Log-Pearson Type III
        skews = [params_dict[dur][0] for dur in params_dict.keys()]
        locs = [params_dict[dur][1] for dur in params_dict.keys()]
        scales = [params_dict[dur][2] for dur in params_dict.keys()]
        
        param_std = {
            'skew_std': np.std(skews),
            'location_std': np.std(locs),
            'scale_std': np.std(scales)
        }
    else:
        param_std = {}
    
    # Average parameter standard deviation as epistemic uncertainty measure
    avg_param_std = np.mean([v for v in param_std.values()])
    
    return {
        'avg_parameter_std': avg_param_std,
        'parameter_variability': param_std
    }


def quantify_total_uncertainty(prediction_metrics, distribution_metrics):
    """
    Combine aleatoric and epistemic uncertainty measures.
    
    Parameters:
        prediction_metrics (dict): Metrics from prediction intervals
        distribution_metrics (dict): Metrics from distribution parameters
        
    Returns:
        dict: Combined uncertainty quantification
    """
    # Aleatoric uncertainty (data/prediction uncertainty)
    aleatoric = prediction_metrics['residual_std']
    
    # Epistemic uncertainty (model/parameter uncertainty)
    epistemic = distribution_metrics['avg_parameter_std']
    
    # Total uncertainty (simplified combination)
    total_uncertainty = np.sqrt(aleatoric**2 + epistemic**2)
    
    return {
        'aleatoric_uncertainty': aleatoric,
        'epistemic_uncertainty': epistemic,
        'total_uncertainty': total_uncertainty
    }


def save_uncertainty_metrics(model_name, uncertainty_dict, output_file='results/uncertainty_analysis.csv'):
    """
    Save uncertainty metrics to CSV file. Updates existing row if model already exists.
    
    Parameters:
        model_name (str): Name of the model/method
        uncertainty_dict (dict): Dictionary of uncertainty metrics
        output_file (str): Path to output CSV file
    """
    # Prepare row data
    row_data = {'model': model_name}
    row_data.update(uncertainty_dict)
    
    # Convert to DataFrame
    df_new = pd.DataFrame([row_data])
    
    # Get the full path
    output_path = os.path.join(os.path.dirname(__file__), '..', output_file)
    
    # Update or create file
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        df_existing = pd.read_csv(output_path)
        # Check if model already exists
        if model_name in df_existing['model'].values:
            # Update existing row
            df_existing = df_existing[df_existing['model'] != model_name]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            # Append new row
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(output_path, index=False)
    print(f"Uncertainty metrics for {model_name} saved to {output_file}")


def analyze_statistical_model_uncertainty(model_name, val_df, durations, model_params, 
                                         return_periods, probabilities, distribution_type):
    """
    Comprehensive uncertainty analysis for statistical IDF models.
    
    Parameters:
        model_name (str): Name of the statistical model
        val_df (DataFrame): Validation data
        durations (list): List of duration column names
        model_params (dict): Fitted distribution parameters
        return_periods (array): Return periods
        probabilities (array): Probabilities
        distribution_type (str): Type of distribution
        
    Returns:
        dict: Complete uncertainty metrics
    """
    all_predictions = []
    all_observations = []
    
    # Collect predictions and observations across all durations
    for dur in durations:
        data_val = val_df[dur].dropna()
        if len(data_val) == 0:
            continue
        
        # Get distribution parameters
        params = model_params[dur]
        
        # Generate predictions for validation data points
        sorted_val_data = np.sort(data_val)[::-1]
        n_val = len(sorted_val_data)
        val_ranks = np.arange(1, n_val + 1)
        val_return_periods = (n_val + 1) / val_ranks
        val_probs = 1 - (1 / val_return_periods)
        
        # Predict intensities using the distribution
        if distribution_type == 'gev':
            shape, loc, scale = params
            predicted_intensities = stats.genextreme.ppf(val_probs, -shape, loc=loc, scale=scale)
        elif distribution_type == 'gumbel':
            loc, scale = params
            predicted_intensities = stats.gumbel_r.ppf(val_probs, loc=loc, scale=scale)
        elif distribution_type == 'weibull':
            shape, loc, scale = params
            predicted_intensities = stats.weibull_min.ppf(val_probs, shape, loc=loc, scale=scale)
        elif distribution_type == 'lognormal':
            shape, loc, scale = params
            predicted_intensities = stats.lognorm.ppf(val_probs, shape, loc=loc, scale=scale)
        elif distribution_type == 'logpearson3':
            skew, loc, scale = params
            predicted_intensities = stats.pearson3.ppf(val_probs, skew, loc=loc, scale=scale)
        else:
            continue
        
        all_predictions.extend(predicted_intensities)
        all_observations.extend(sorted_val_data)
    
    if len(all_predictions) == 0:
        print(f"Warning: No predictions available for {model_name}")
        return {}
    
    # Calculate prediction interval metrics
    prediction_metrics = calculate_prediction_intervals(
        np.array(all_predictions), 
        np.array(all_observations)
    )
    
    # Calculate distribution parameter uncertainty
    distribution_metrics = calculate_distribution_uncertainty(
        model_params, 
        distribution_type, 
        return_periods, 
        val_df
    )
    
    # Quantify total uncertainty
    total_metrics = quantify_total_uncertainty(prediction_metrics, distribution_metrics)
    
    # Combine all metrics
    uncertainty_metrics = {
        **prediction_metrics,
        **total_metrics
    }
    
    # Save to CSV
    save_uncertainty_metrics(model_name, uncertainty_metrics)
    
    return uncertainty_metrics
