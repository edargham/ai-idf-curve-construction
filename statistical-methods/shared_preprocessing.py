import os
import pandas as pd
import numpy as np


def load_annual_max_intensity(filename="annual_max_intensity.csv"):
    """
    Load the processed annual maximum intensity data.
    
    Parameters:
        filename (str): Name of the CSV file containing annual max intensity data
        
    Returns:
        pd.DataFrame: DataFrame with years as index and duration columns
    """
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Annual max intensity file not found at: {file_path}\n"
            f"Please run idf-precip-aggregator.py first to generate the data."
        )
    
    df = pd.read_csv(file_path, index_col=0)
    df.index = df.index.astype(int)
    
    return df


def split_train_validation(df, train_years=None, val_years=None):
    """
    Split annual maximum intensity data into training and validation sets.
    
    Parameters:
        df (pd.DataFrame): Annual maximum intensity dataframe with years as index
        train_years (list): List of years for training (default: 1998-2018)
        val_years (list): List of years for validation (default: 2019-2025)
        
    Returns:
        tuple: (train_df, val_df) DataFrames for training and validation
    """
    if train_years is None:
        train_years = list(range(1998, 2019))
    
    if val_years is None:
        val_years = list(range(2019, 2026))
    
    # Ensure index is integer year
    df_copy = df.copy()
    df_copy.index = df_copy.index.astype(int)
    
    train_df = df_copy.loc[df_copy.index.isin(train_years)]
    val_df = df_copy.loc[df_copy.index.isin(val_years)]
    
    if train_df.empty:
        print(
            f"Warning: Training dataframe for years {min(train_years)}-{max(train_years)} "
            f"is empty. Check your data range."
        )
    
    if val_df.empty:
        print(
            f"Warning: Validation dataframe for years {min(val_years)}-{max(val_years)} "
            f"is empty. Check your data range."
        )
    
    return train_df, val_df


def get_duration_configs():
    """
    Get standard duration configurations used across the analysis.
    
    Returns:
        tuple: (durations, duration_hours, duration_labels)
    """
    durations = [
        "5mns", "10mns", "15mns", "30mns", "1h", "90min", "2h", "3h",
        "6h", "12h", "15h", "18h", "24h"
    ]
    
    duration_hours = [
        5/60, 10/60, 15/60, 30/60, 1, 1.5, 2, 3, 6, 12, 15, 18, 24
    ]
    
    duration_labels = [
        "5 mins", "10 mins", "15 mins", "30 mins", "60 mins", "90 mins",
        "120 mins", "180 mins", "360 mins", "720 mins", "900 mins",
        "1080 mins", "1440 mins"
    ]
    
    return durations, duration_hours, duration_labels


def get_return_periods():
    """
    Get standard return periods used in IDF analysis.
    
    Returns:
        tuple: (return_periods, probabilities)
    """
    return_periods = np.array([2, 5, 10, 25, 50, 100])
    probabilities = 1 - 1 / return_periods
    
    return return_periods, probabilities
