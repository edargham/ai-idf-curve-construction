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


def compute_mc_dropout_uncertainty(model, X_tensor, device, n_samples=30):
    """
    Compute epistemic uncertainty using Monte Carlo Dropout.
    
    Parameters:
        model: PyTorch model with dropout layers
        X_tensor: Input tensor
        device: Device (CPU/CUDA/MPS)
        n_samples: Number of MC samples
        
    Returns:
        tuple: (mean_predictions, epistemic_std)
    """
    try:
        import torch
        
        model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = model(X_tensor).cpu().numpy().flatten()
                predictions.append(output)
        
        model.eval()  # Disable dropout
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        epistemic_std = predictions.std(axis=0).mean()
        
        return mean_pred, epistemic_std
    except Exception as e:
        print(f"Warning: MC Dropout failed: {e}")
        return None, None


def compute_bootstrap_uncertainty(model, X_train, y_train, X_val, n_bootstrap=20):
    """
    Compute epistemic uncertainty using bootstrap resampling for non-neural models.
    
    Parameters:
        model: Scikit-learn model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        tuple: (mean_predictions, epistemic_std)
    """
    try:
        from sklearn.utils import resample
        
        predictions = []
        n_samples = len(X_train)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = resample(range(n_samples), n_samples=n_samples, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Clone and fit model on bootstrap sample
            from sklearn.base import clone
            boot_model = clone(model)
            boot_model.fit(X_boot, y_boot)
            
            # Predict on validation set
            pred = boot_model.predict(X_val)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        epistemic_std = predictions.std(axis=0).mean()
        
        return mean_pred, epistemic_std
    except Exception as e:
        print(f"Warning: Bootstrap uncertainty estimation failed: {e}")
        return None, None


def analyze_ai_model_uncertainty(model_name, predictions, observations, epistemic_std=None, 
                                 model=None, X_val=None, device=None, scaler_y=None,
                                 use_mc_dropout=False, n_mc_samples=30,
                                 use_bootstrap=False, X_train=None, y_train=None, n_bootstrap=20):
    """
    Comprehensive uncertainty analysis for AI-based IDF models.
    
    Parameters:
        model_name (str): Name of the AI model
        predictions (array): Model predictions
        observations (array): Observed values
        epistemic_std (float): Optional epistemic uncertainty measure
        model: PyTorch model (for MC Dropout) or sklearn model (for bootstrap)
        X_val: Validation input tensor/array
        device: Computation device (for MC Dropout)
        scaler_y: Scaler for inverse transform
        use_mc_dropout (bool): Whether to use MC Dropout
        n_mc_samples (int): Number of MC Dropout samples
        use_bootstrap (bool): Whether to use bootstrap (for sklearn models)
        X_train: Training features (for bootstrap)
        y_train: Training targets (for bootstrap)
        n_bootstrap (int): Number of bootstrap samples
        
    Returns:
        dict: Complete uncertainty metrics
    """
    # Compute epistemic uncertainty via MC Dropout if requested
    if use_mc_dropout and model is not None and X_val is not None:
        print(f"  Computing epistemic uncertainty via MC Dropout ({n_mc_samples} samples)...")
        mc_pred_scaled, mc_epistemic_std_scaled = compute_mc_dropout_uncertainty(
            model, X_val, device, n_mc_samples
        )
        
        if mc_epistemic_std_scaled is not None and scaler_y is not None:
            # Transform epistemic std from scaled space to original space
            # Approximate transformation for std
            epistemic_std = mc_epistemic_std_scaled * scaler_y.scale_[0] if hasattr(scaler_y, 'scale_') else mc_epistemic_std_scaled
        elif mc_epistemic_std_scaled is not None:
            epistemic_std = mc_epistemic_std_scaled
    
    # Compute epistemic uncertainty via bootstrap if requested
    elif use_bootstrap and model is not None and X_train is not None and y_train is not None and X_val is not None:
        print(f"  Computing epistemic uncertainty via Bootstrap ({n_bootstrap} samples)...")
        boot_pred_scaled, boot_epistemic_std_scaled = compute_bootstrap_uncertainty(
            model, X_train, y_train, X_val, n_bootstrap
        )
        
        if boot_epistemic_std_scaled is not None and scaler_y is not None:
            # Transform epistemic std from scaled space to original space
            epistemic_std = boot_epistemic_std_scaled * scaler_y.scale_[0] if hasattr(scaler_y, 'scale_') else boot_epistemic_std_scaled
        elif boot_epistemic_std_scaled is not None:
            epistemic_std = boot_epistemic_std_scaled
    
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
