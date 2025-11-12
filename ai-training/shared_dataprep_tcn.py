import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from split_utils import build_train_val
from shared_io import shared_preprocessing

import numpy as np


class TensorRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
    # This is superior to StandardScaler for neural networks because:
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
