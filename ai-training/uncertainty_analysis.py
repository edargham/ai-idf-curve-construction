"""
Uncertainty Analysis for AI-based IDF Models

This module provides functions to quantify uncertainty in machine learning and deep learning
IDF curve predictions, including prediction variance and model confidence.
"""

import numpy as np
import pandas as pd
import os


def calculate_prediction_intervals(predictions, observations, confidence_level=0.95):
    """
    Calculate prediction intervals for AI models.
    
    Parameters:
        predictions (array): Model predictions
        observations (array): Observed values
        confidence_level (float): Confidence level (default 0.95 for 95% CI)
        
    Returns:
        dict: Dictionary containing uncertainty metrics
    """
    from scipy import stats
    
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


def calculate_epistemic_uncertainty(predictions_list):
    """
    Calculate epistemic uncertainty from multiple predictions (e.g., from ensemble or dropout).
    
    Parameters:
        predictions_list (list of arrays): List of prediction arrays from multiple runs
        
    Returns:
        dict: Epistemic uncertainty metrics
    """
    if len(predictions_list) == 0:
        return {
            'prediction_variance': np.nan,
            'prediction_std': np.nan,
            'epistemic_uncertainty': np.nan
        }
    
    # Stack predictions
    predictions_array = np.array(predictions_list)
    
    # Variance across predictions (epistemic uncertainty)
    prediction_variance = np.var(predictions_array, axis=0).mean()
    prediction_std = np.std(predictions_array, axis=0).mean()
    
    return {
        'prediction_variance': prediction_variance,
        'prediction_std': prediction_std,
        'epistemic_uncertainty': prediction_std
    }


def quantify_total_uncertainty(prediction_metrics, epistemic_std=None):
    """
    Combine aleatoric and epistemic uncertainty measures.
    
    Parameters:
        prediction_metrics (dict): Metrics from prediction intervals
        epistemic_std (float): Epistemic uncertainty (if available)
        
    Returns:
        dict: Combined uncertainty quantification
    """
    # Aleatoric uncertainty (data/prediction uncertainty)
    aleatoric = prediction_metrics['residual_std']
    
    # Epistemic uncertainty (model uncertainty)
    if epistemic_std is None or np.isnan(epistemic_std):
        # Use a proxy: assume epistemic is a fraction of aleatoric
        epistemic = aleatoric * 0.5  # Conservative estimate
    else:
        epistemic = epistemic_std
    
    # Total uncertainty
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


def analyze_ai_model_uncertainty(model_name, predictions, observations, epistemic_std=None):
    """
    Comprehensive uncertainty analysis for AI-based IDF models.
    
    Parameters:
        model_name (str): Name of the AI model
        predictions (array): Model predictions
        observations (array): Observed values
        epistemic_std (float): Optional epistemic uncertainty measure
        
    Returns:
        dict: Complete uncertainty metrics
    """
    # Calculate prediction interval metrics
    prediction_metrics = calculate_prediction_intervals(
        predictions, 
        observations
    )
    
    # Quantify total uncertainty
    total_metrics = quantify_total_uncertainty(prediction_metrics, epistemic_std)
    
    # Combine all metrics
    uncertainty_metrics = {
        **prediction_metrics,
        **total_metrics
    }
    
    # Save to CSV
    save_uncertainty_metrics(model_name, uncertainty_metrics)
    
    return uncertainty_metrics
