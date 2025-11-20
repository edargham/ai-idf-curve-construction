"""
Shared Optuna hyperparameter tuning framework for all models.
Maximizes Nash-Sutcliffe Efficiency (NSE) on validation data.

This module provides a unified interface for tuning:
- SVM (scikit-learn)
- ANN (PyTorch MLP)
- TCN (PyTorch Temporal Convolutional Network)
- TCAN (PyTorch TCN with Attention)

Key design principles:
1. No data leakage: strict train/validation split
2. Maximize NSE as primary objective
3. Apple Silicon (MPS) support for PyTorch models
4. Consistent API across all model types
"""

import os
import random
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

from shared_metrics import nash_sutcliffe_efficiency, squared_pearson_r2 as r2_score
from shared_dataprep_tcn import TensorRegressionDataset

# Reproducibility seed
SEED = 42


def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def get_device():
    """Get optimal PyTorch device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def evaluate_predictions(y_true, y_pred):
    """
    Compute evaluation metrics: NSE, RMSE, MAE, R².
    
    Args:
        y_true: Ground truth values (numpy array or tensor)
        y_pred: Predicted values (numpy array or tensor)
    
    Returns:
        dict with keys: 'nse', 'rmse', 'mae', 'r2'
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Ensure same length
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    nse = nash_sutcliffe_efficiency(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'nse': nse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


# ============================================================================
# SVM Objective Function
# ============================================================================

def create_svm_objective(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler_y):
    """
    Create Optuna objective function for SVM hyperparameter tuning.
    
    Args:
        X_train_scaled: Scaled training features (numpy array)
        y_train_scaled: Scaled training targets (numpy array)
        X_val_scaled: Scaled validation features (numpy array)
        y_val_scaled: Scaled validation targets (numpy array)
        scaler_y: Scaler for inverse transforming targets (CRITICAL for proper NSE on original scale)
    
    Returns:
        objective function for Optuna
    """
    def objective(trial):
        # Suggest hyperparameters
        C = trial.suggest_float("C", 0.01, 100.0, log=True)
        epsilon = trial.suggest_float("epsilon", 0.001, 0.5, log=True)
        gamma = trial.suggest_float("gamma", 0.001, 10.0, log=True)
        kernel = trial.suggest_categorical("kernel", ["rbf"])
        
        # Train model
        model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)
        model.fit(X_train_scaled, y_train_scaled)
        
        # Predict on validation set (scaled)
        y_pred_scaled = model.predict(X_val_scaled)
        
        # Inverse transform to original scale for fair metric comparison
        y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_val_log = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
        
        # Transform from log-space to original intensity (mm/hr)
        # Clip to prevent overflow
        y_pred_log = np.clip(y_pred_log, -10, 15)
        y_val_log = np.clip(y_val_log, -10, 15)
        
        y_pred_intensity = np.exp(y_pred_log)
        y_val_intensity = np.exp(y_val_log)
        
        # Evaluate on original scale (maximize NSE)
        metrics = evaluate_predictions(y_val_intensity, y_pred_intensity)
        
        # Log additional metrics
        trial.set_user_attr("rmse", metrics['rmse'])
        trial.set_user_attr("mae", metrics['mae'])
        trial.set_user_attr("r2", metrics['r2'])
        
        # Return NSE (Optuna will maximize this)
        return metrics['nse']
    
    return objective


# ============================================================================
# PyTorch Training Utilities
# ============================================================================

def train_pytorch_epoch(model, loader, optimizer, criterion, device):
    """Train PyTorch model for one epoch."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def evaluate_pytorch_model(model, loader, device, scaler_y=None):
    """
    Evaluate PyTorch model on a dataset.
    
    Args:
        model: PyTorch model
        loader: DataLoader
        device: torch device
        scaler_y: Optional scaler to inverse transform predictions (transforms from scaled to log-space)
    
    Returns:
        dict with metrics (nse, rmse, mae, r2) computed on ORIGINAL SCALE (mm/hr)
    """
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            all_preds.append(out.cpu())
            all_trues.append(yb)
    
    preds = torch.cat(all_preds).numpy().flatten()
    trues = torch.cat(all_trues).numpy().flatten()
    
    # Inverse transform if scaler provided (scaled log-intensity → log-intensity)
    if scaler_y is not None:
        preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
        trues = scaler_y.inverse_transform(trues.reshape(-1, 1)).flatten()
        
        # Transform from log-space to original intensity (mm/hr)
        # Clip to prevent overflow: log-intensity typically ranges from -2 to 8
        # exp(20) ≈ 485 million mm/hr is physically unrealistic (cap at exp(15) ≈ 3.3M mm/hr)
        preds = np.clip(preds, -10, 15)  # Prevent numerical overflow
        trues = np.clip(trues, -10, 15)
        
        preds = np.exp(preds)
        trues = np.exp(trues)
    
    return evaluate_predictions(trues, preds)


def train_sequential_epoch(model, loader, optimizer, criterion, device):
    """Train sequential model for one epoch."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)  # [batch, 13, seq_len], [batch, 13, 1]
        optimizer.zero_grad()
        out = model(xb)  # [batch, 13, 1]
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def evaluate_sequential_model(model, loader, device, scaler_y=None):
    """
    Evaluate sequential model on a dataset.
    
    Args:
        model: Sequential TCN/TCAN model
        loader: DataLoader returning ([batch, 13, seq_len], [batch, 13, 1])
        device: torch device
        scaler_y: Optional scaler to inverse transform predictions
    
    Returns:
        dict with metrics (nse, rmse, mae, r2) for base log intensity predictions
    """
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)  # [batch, 13, 1]
            all_preds.append(out.cpu())
            all_trues.append(yb)
    
    preds = torch.cat(all_preds).numpy()  # [N, 13, 1]
    trues = torch.cat(all_trues).numpy()  # [N, 13, 1]
    
    # Inverse transform if scaler provided
    if scaler_y is not None:
        # Flatten, transform, reshape
        preds_flat = preds.reshape(-1, 1)
        trues_flat = trues.reshape(-1, 1)
        
        preds_flat = scaler_y.inverse_transform(preds_flat)
        trues_flat = scaler_y.inverse_transform(trues_flat)
        
        preds = preds_flat.reshape(preds.shape)
        trues = trues_flat.reshape(trues.shape)
    
    # Flatten all predictions and targets for metric computation
    preds_flat = preds.flatten()
    trues_flat = trues.flatten()
    
    return evaluate_predictions(trues_flat, preds_flat)


# ============================================================================
# ANN (MLP) Objective Function
# ============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron for regression."""
    def __init__(self, input_dim, hidden_size, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def create_ann_objective(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
                         scaler_y, device, max_epochs=300):
    """
    Create Optuna objective function for ANN hyperparameter tuning.
    
    Args:
        X_train_scaled: Scaled training features
        y_train_scaled: Scaled training targets
        X_val_scaled: Scaled validation features
        y_val_scaled: Scaled validation targets
        scaler_y: Scaler for inverse transforming targets (CRITICAL for proper NSE)
        device: torch device
        max_epochs: Maximum training epochs per trial
    
    Returns:
        objective function for Optuna
    """
    def objective(trial):
        set_seed(SEED)
        
        # Suggest hyperparameters - AGGRESSIVE SEARCH SPACE FOR NSE > 0.95
        hidden_size = trial.suggest_categorical("hidden_size", 
                                                 [512, 768, 1024, 1536, 2048, 3072, 4096])
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)  # Wider range, higher ceiling
        dropout = trial.suggest_float("dropout", 0.0, 0.3)  # Allow NO dropout
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])  # Smaller + larger
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)  # Wider range
        
        # Create model
        model = MLP(input_dim=X_train_scaled.shape[1], 
                    hidden_size=hidden_size,
                    dropout=dropout).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15
        )
        
        # Create data loaders
        train_gen = torch.Generator()
        train_gen.manual_seed(SEED)
        
        train_ds = TensorRegressionDataset(X_train_scaled, y_train_scaled)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=train_gen)
        
        val_ds = TensorRegressionDataset(X_val_scaled, y_val_scaled)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Early stopping - INCREASED PATIENCE
        best_val_nse = -np.inf
        patience = 75
        patience_counter = 0
        
        # Training loop
        for epoch in range(max_epochs):
            train_pytorch_epoch(model, train_loader, optimizer, criterion, device)
            
            # Evaluate on validation - WITH INVERSE TRANSFORM FOR PROPER NSE
            metrics = evaluate_pytorch_model(model, val_loader, device, scaler_y=scaler_y)
            val_nse = metrics['nse']
            
            # Update scheduler
            scheduler.step(val_nse)
            
            # Early stopping check
            if val_nse > best_val_nse:
                best_val_nse = val_nse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Report intermediate value for pruning
            trial.report(val_nse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Final evaluation - WITH INVERSE TRANSFORM
        final_metrics = evaluate_pytorch_model(model, val_loader, device, scaler_y=scaler_y)
        
        # Log metrics
        trial.set_user_attr("rmse", final_metrics['rmse'])
        trial.set_user_attr("mae", final_metrics['mae'])
        trial.set_user_attr("r2", final_metrics['r2'])
        trial.set_user_attr("epochs_trained", epoch + 1)
        
        return final_metrics['nse']
    
    return objective


# ============================================================================
# TCN Components and Objective
# ============================================================================

class TemporalBlock(nn.Module):
    """Simplified temporal block with single-scale convolution and residual connections.
    
    Uses a straightforward architecture focused on learning the temporal patterns
    without unnecessary complexity that can hinder convergence.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.01):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) // 2  # same padding
        
        # Two conv layers with batch norm
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()
        
    def forward(self, x):
        # Save input for residual
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        # Add residual and activate
        return self.relu_out(out + identity)


class SequentialTCN(nn.Module):
    """
    Temporal Convolutional Network for sequential IDF prediction.
    
    Takes multi-channel time-series input [batch, 13_durations, seq_len]
    Outputs base log intensity predictions [batch, 13_durations, 1]
    """
    def __init__(self, num_channels=13, seq_len=256, num_filters=64, 
                 kernel_size=3, dropout=0.2, num_levels=4):
        super(SequentialTCN, self).__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.num_durations = 13
        self.num_return_periods = 1  # Now predicting base intensities only
        
        # TCN layers - process each duration channel
        layers = []
        in_ch = num_channels
        for i in range(num_levels):
            out_ch = num_filters * (2 ** min(i, 2))  # Cap at 4x for memory
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch
        
        self.tcn_network = nn.Sequential(*layers)
        
        # Global pooling to reduce temporal dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final channel count
        final_channels = num_filters * (2 ** min(num_levels - 1, 2))
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(final_channels, final_channels * 2)
        self.bn1 = nn.BatchNorm1d(final_channels * 2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout * 0.5)
        
        self.fc2 = nn.Linear(final_channels * 2, final_channels)
        self.bn2 = nn.BatchNorm1d(final_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout * 0.5)
        
        # Output layer: 78 values (13 durations × 6 return periods)
        self.fc_out = nn.Linear(final_channels, self.num_durations * self.num_return_periods)
    
    def forward(self, x):
        """
        Args:
            x: [batch, 13, seq_len] - multi-channel time-series
        
        Returns:
            [batch, 13, 1] - base log intensity predictions
        """
        # TCN feature extraction
        x = self.tcn_network(x)  # [batch, final_channels, reduced_seq]
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # [batch, final_channels]
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Output
        x = self.fc_out(x)  # [batch, 13]
        
        # Reshape to [batch, 13, 1]
        return x.view(-1, self.num_durations, self.num_return_periods)


class SequentialTCAN(nn.Module):
    """
    Temporal Convolutional Attention Network for sequential IDF prediction.
    
    Takes multi-channel time-series input [batch, 13_durations, seq_len]
    Outputs base log intensity predictions [batch, 13_durations, 1]
    """
    def __init__(self, num_channels=13, seq_len=256, num_filters=64, 
                 kernel_size=3, dropout=0.2, num_levels=4, 
                 attn_heads=8, attn_dropout=0.1):
        super(SequentialTCAN, self).__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.num_durations = 13
        self.num_return_periods = 1  # Now predicting base intensities only
        
        # TCN layers with attention - process each duration channel
        layers = []
        in_ch = num_channels
        for i in range(num_levels):
            out_ch = num_filters * (2 ** min(i, 2))  # Cap at 4x for memory
            layers.append(TemporalBlockWithAttention(
                in_ch, out_ch, kernel_size, dropout, 
                attn_heads=attn_heads, attn_dropout=attn_dropout
            ))
            in_ch = out_ch
        
        self.tcan_network = nn.Sequential(*layers)
        
        # Global pooling to reduce temporal dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final channel count
        final_channels = num_filters * (2 ** min(num_levels - 1, 2))
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(final_channels, final_channels * 2)
        self.bn1 = nn.BatchNorm1d(final_channels * 2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout * 0.5)
        
        self.fc2 = nn.Linear(final_channels * 2, final_channels)
        self.bn2 = nn.BatchNorm1d(final_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout * 0.5)
        
        # Output layer: 13 values (13 durations × 1 base intensity)
        self.fc_out = nn.Linear(final_channels, self.num_durations * self.num_return_periods)
    
    def forward(self, x):
        """
        Args:
            x: [batch, 13, seq_len] - multi-channel time-series
        
        Returns:
            [batch, 13, 1] - base log intensity predictions
        """
        # TCAN feature extraction
        x = self.tcan_network(x)  # [batch, final_channels, reduced_seq]
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # [batch, final_channels]
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Output
        x = self.fc_out(x)  # [batch, 13]
        
        # Reshape to [batch, 13, 1]
        return x.view(-1, self.num_durations, self.num_return_periods)
        
        # Output
        x = self.fc_out(x)  # [batch, 78]
        
        # Reshape to [batch, 13, 6]
        return x.view(-1, self.num_durations, self.num_return_periods)


class TCN(nn.Module):
    """Temporal Convolutional Network for regression."""
    def __init__(self, seq_len=2, num_filters=144, kernel_size=3, dropout=0.01, num_levels=2):
        super(TCN, self).__init__()
        self.seq_len = seq_len
        
        # Simplified 2-level architecture with gradual channel expansion
        layers = []
        in_channels = 1
        for i in range(num_levels):
            out_channels = num_filters * (2 ** i)
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, dropout
            ))
            in_channels = out_channels
        
        self.network = nn.Sequential(*layers)
        
        # Global average pooling for consistent output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Simplified fully connected layers with minimal dropout
        final_channels = num_filters * (2 ** (num_levels - 1))
        self.fc1 = nn.Linear(final_channels, final_channels * 2)
        self.bn_fc1 = nn.BatchNorm1d(final_channels * 2)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout * 0.1)
        
        self.fc2 = nn.Linear(final_channels * 2, final_channels)
        self.bn_fc2 = nn.BatchNorm1d(final_channels)
        self.relu_fc2 = nn.ReLU()
        
        self.fc3 = nn.Linear(final_channels, 1)

    def forward(self, x):
        # x shape: (batch_size, 1, seq_len) or (batch_size, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # add channel dimension -> (batch, 1, seq_len)
        
        # Temporal feature extraction
        x = self.network(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        
        x = self.fc3(x)
        
        return x


def create_tcn_objective(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                         scaler_y, device, max_epochs=400):
    """
    Create Optuna objective function for TCN hyperparameter tuning.
    
    Args:
        X_train_scaled: Scaled training features
        y_train_scaled: Scaled training targets
        X_val_scaled: Scaled validation features
        y_val_scaled: Scaled validation targets
        scaler_y: Scaler for inverse transforming targets (CRITICAL for proper NSE)
        device: torch device
        max_epochs: Maximum training epochs per trial
    
    Returns:
        objective function for Optuna
    """
    def objective(trial):
        set_seed(SEED)
        
        # Suggest hyperparameters - ALIGNED WITH SUCCESSFUL OLD VERSION
        # Old version used: hidden_candidates = [144, 160, 176, 192]
        # lr=8e-4, weight_decay=1e-6, batch_size=64, epochs=300, kernel_size=3, dropout=0.01, num_levels=2
        num_filters = trial.suggest_categorical("num_filters", [128, 144, 160, 176, 192, 208, 224, 256])
        
        # Constrain kernel_size based on sequence length to avoid padding issues
        # For seq_len=2, max safe kernel_size is 3
        seq_len = X_train_scaled.shape[1]
        max_kernel = min(5, seq_len + 1)  # Conservative: kernel can be at most seq_len + 1
        kernel_size = trial.suggest_int("kernel_size", 3, max_kernel) if max_kernel >= 3 else 3
        
        dropout = trial.suggest_float("dropout", 0.005, 0.05)  # Centered around 0.01
        num_levels = trial.suggest_int("num_levels", 2, 4)  # Start with 2 (original)
        lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)  # Centered around 8e-4
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # Default 64
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-5, log=True)  # Centered around 1e-6
        
        # Create model
        model = TCN(seq_len=seq_len,
                    num_filters=num_filters,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    num_levels=num_levels).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        # More aggressive scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=20, min_lr=1e-7
        )
        
        # Create data loaders
        train_gen = torch.Generator()
        train_gen.manual_seed(SEED)
        
        train_ds = TensorRegressionDataset(X_train_scaled, y_train_scaled)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=train_gen)
        
        val_ds = TensorRegressionDataset(X_val_scaled, y_val_scaled)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Early stopping - MATCH OLD VERSION TRAINING REGIME
        best_val_nse = -np.inf
        patience = 80  # Reasonable for focused search space
        patience_counter = 0
        
        # Training loop
        for epoch in range(max_epochs):
            train_pytorch_epoch(model, train_loader, optimizer, criterion, device)
            
            # Evaluate on validation - WITH INVERSE TRANSFORM FOR PROPER NSE
            try:
                metrics = evaluate_pytorch_model(model, val_loader, device, scaler_y=scaler_y)
                val_nse = metrics['nse']
                
                # Check for numerical issues (divergence)
                if not np.isfinite(val_nse):
                    raise optuna.TrialPruned()
                    
            except (ValueError, RuntimeWarning, FloatingPointError):
                # Model diverged (overflow/underflow) - prune trial
                raise optuna.TrialPruned()
            
            # Update scheduler based on validation NSE
            scheduler.step(val_nse)
            
            # Early stopping check
            if val_nse > best_val_nse:
                best_val_nse = val_nse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Report intermediate value for pruning
            trial.report(val_nse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Final evaluation - WITH INVERSE TRANSFORM
        final_metrics = evaluate_pytorch_model(model, val_loader, device, scaler_y=scaler_y)
        
        # Log metrics
        trial.set_user_attr("rmse", final_metrics['rmse'])
        trial.set_user_attr("mae", final_metrics['mae'])
        trial.set_user_attr("r2", final_metrics['r2'])
        trial.set_user_attr("epochs_trained", epoch + 1)
        
        return final_metrics['nse']
    
    return objective


# ============================================================================
# TCAN Components and Objective
# ============================================================================

class MHSA1d(nn.Module):
    """Multi-Head Self-Attention over 1D feature maps.
    
    Treats the length dimension (L) as the token axis and the channel dimension (C)
    as the embedding size. Input/Output shape: (B, C, L).
    """
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.05):
        super().__init__()
        # Ensure num_heads divides channels; if not, fallback to the largest divisor <= num_heads
        nh = max(1, min(num_heads, channels))
        for h in range(min(num_heads, channels), 0, -1):
            if channels % h == 0:
                nh = h
                break
        self.num_heads = nh
        self.embed_dim = channels
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, L, C)
        x_seq = x.permute(0, 2, 1)
        x_norm = self.ln1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x_res = x_seq + self.dropout(attn_out)
        x_res = self.ln2(x_res)
        # Back to (B, C, L)
        return x_res.permute(0, 2, 1)


class TemporalBlockWithAttention(nn.Module):
    """Simplified temporal block with residual convs and optional self-attention."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.01, attn_heads: int = 0, attn_dropout: float = 0.05):
        super(TemporalBlockWithAttention, self).__init__()

        padding = (kernel_size - 1) // 2  # same padding

        # Two conv layers with batch norm
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

        # Optional attention after residual fusion
        self.attn = MHSA1d(out_channels, num_heads=attn_heads, dropout=attn_dropout) if attn_heads and attn_heads > 0 else None

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu_out(out + identity)

        # Self-attention refinement (keeps shape)
        if self.attn is not None:
            out = self.attn(out)

        return out


class TCAN(nn.Module):
    """Temporal Convolutional Attention Network for regression."""
    def __init__(self, seq_len=2, num_filters=144, kernel_size=3, dropout=0.01, 
                 num_levels=2, attn_heads=8, attn_dropout=0.05):
        super(TCAN, self).__init__()
        self.seq_len = seq_len
        
        # Simplified 2-level architecture with gradual channel expansion
        layers = []
        in_channels = 1
        for i in range(num_levels):
            out_channels = num_filters * (2 ** i)
            layers.append(TemporalBlockWithAttention(
                in_channels, out_channels, kernel_size, dropout, attn_heads=attn_heads, attn_dropout=attn_dropout
            ))
            in_channels = out_channels
        
        self.network = nn.Sequential(*layers)
        
        # Global average pooling for consistent output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Simplified fully connected layers with minimal dropout
        final_channels = num_filters * (2 ** (num_levels - 1))
        self.fc1 = nn.Linear(final_channels, final_channels * 2)
        self.bn_fc1 = nn.BatchNorm1d(final_channels * 2)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout * 0.1)
        
        self.fc2 = nn.Linear(final_channels * 2, final_channels)
        self.bn_fc2 = nn.BatchNorm1d(final_channels)
        self.relu_fc2 = nn.ReLU()
        
        self.fc3 = nn.Linear(final_channels, 1)

    def forward(self, x):
        # x shape: (batch_size, 1, seq_len) or (batch_size, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # add channel dimension -> (batch, 1, seq_len)
        
        # Temporal feature extraction
        x = self.network(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        
        x = self.fc3(x)
        
        return x


def create_tcan_objective(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                          scaler_y, device, max_epochs=400):
    """
    Create Optuna objective function for TCAN hyperparameter tuning.
    
    Args:
        X_train_scaled: Scaled training features
        y_train_scaled: Scaled training targets
        X_val_scaled: Scaled validation features
        y_val_scaled: Scaled validation targets
        scaler_y: Scaler for inverse transforming targets (CRITICAL for proper NSE)
        device: torch device
        max_epochs: Maximum training epochs per trial
    
    Returns:
        objective function for Optuna
    """
    def objective(trial):
        set_seed(SEED)
        
        # Suggest hyperparameters - ALIGNED WITH SUCCESSFUL OLD TCAN VERSION
        # Old version used: hidden_candidates = [144, 160, 176, 192]
        # attn_heads=8, attn_dropout=0.05, lr=8e-4, batch_size=64, kernel_size=3, dropout=0.01, num_levels=2
        attn_heads = trial.suggest_categorical("attn_heads", [4, 8, 16])  # Centered around 8
        
        # num_filters must be divisible by attn_heads for MultiheadAttention
        # Focus on moderate sizes that worked in old version
        all_filters = [128, 144, 160, 176, 192, 208, 224, 256, 288, 320, 352, 384]
        
        # Filter based on divisibility by attn_heads
        valid_filters = [f for f in all_filters if f % attn_heads == 0]
        num_filters = trial.suggest_categorical("num_filters", valid_filters)
        
        # Constrain kernel_size based on sequence length to avoid padding issues
        seq_len = X_train_scaled.shape[1]
        max_kernel = min(5, seq_len + 1)  # Conservative: kernel can be at most seq_len + 1
        kernel_size = trial.suggest_int("kernel_size", 3, max_kernel) if max_kernel >= 3 else 3
        
        dropout = trial.suggest_float("dropout", 0.005, 0.05)  # Centered around 0.01
        num_levels = trial.suggest_int("num_levels", 2, 4)  # Start with 2 (original)
        attn_dropout = trial.suggest_float("attn_dropout", 0.02, 0.1)  # Centered around 0.05
        lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)  # Centered around 8e-4
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # Default 64
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-5, log=True)  # Centered around 1e-6
        
        # Create model
        model = TCAN(seq_len=seq_len,
                     num_filters=num_filters,
                     kernel_size=kernel_size,
                     dropout=dropout,
                     num_levels=num_levels,
                     attn_heads=attn_heads,
                     attn_dropout=attn_dropout).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        # Use ReduceLROnPlateau for adaptive learning based on validation performance
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=25, min_lr=1e-7
        )
        
        # Create data loaders
        train_gen = torch.Generator()
        train_gen.manual_seed(SEED)
        
        train_ds = TensorRegressionDataset(X_train_scaled, y_train_scaled)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=train_gen)
        
        val_ds = TensorRegressionDataset(X_val_scaled, y_val_scaled)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Early stopping - MATCH OLD VERSION TRAINING REGIME
        best_val_nse = -np.inf
        patience = 80  # Reasonable for focused search space
        patience_counter = 0
        
        # Training loop
        for epoch in range(max_epochs):
            train_pytorch_epoch(model, train_loader, optimizer, criterion, device)
            
            # Evaluate on validation - WITH INVERSE TRANSFORM FOR PROPER NSE
            try:
                metrics = evaluate_pytorch_model(model, val_loader, device, scaler_y=scaler_y)
                val_nse = metrics['nse']
                
                # Check for numerical issues (divergence)
                if not np.isfinite(val_nse):
                    raise optuna.TrialPruned()
                    
            except (ValueError, RuntimeWarning, FloatingPointError):
                # Model diverged (overflow/underflow) - prune trial
                raise optuna.TrialPruned()
            
            # Update scheduler based on validation NSE
            scheduler.step(val_nse)
            
            # Early stopping check
            if val_nse > best_val_nse:
                best_val_nse = val_nse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Report intermediate value for pruning
            trial.report(val_nse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Final evaluation - WITH INVERSE TRANSFORM
        final_metrics = evaluate_pytorch_model(model, val_loader, device, scaler_y=scaler_y)
        
        # Log metrics
        trial.set_user_attr("rmse", final_metrics['rmse'])
        trial.set_user_attr("mae", final_metrics['mae'])
        trial.set_user_attr("r2", final_metrics['r2'])
        trial.set_user_attr("epochs_trained", epoch + 1)
        
        return final_metrics['nse']
    
    return objective


# ============================================================================
# High-Level Tuning Interface
# ============================================================================

def run_optuna_study(objective, model_name, n_trials=100, direction="maximize"):
    """
    Run Optuna hyperparameter optimization study.
    
    Args:
        objective: Optuna objective function
        model_name: Name of model (for study name and results)
        n_trials: Number of trials to run
        direction: "maximize" or "minimize"
    
    Returns:
        study: Optuna study object with results
    """
    set_seed(SEED)
    
    # Create study with TPE sampler and median pruner
    sampler = TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    
    study = optuna.create_study(
        study_name=f"{model_name}_optuna_study",
        direction=direction,
        sampler=sampler,
        pruner=pruner
    )
    
    # Run optimization
    print(f"\n{'='*80}")
    print(f"Starting Optuna hyperparameter tuning for {model_name}")
    print(f"Objective: {direction.upper()} NSE (Nash-Sutcliffe Efficiency)")
    print(f"Trials: {n_trials}")
    print(f"{'='*80}\n")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Optuna Study Complete: {model_name}")
    print(f"{'='*80}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best NSE: {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    if study.best_trial.user_attrs:
        print("\nBest trial metrics:")
        for key, value in study.best_trial.user_attrs.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    print(f"{'='*80}\n")
    
    return study


def save_optuna_results(study, model_name, output_dir="../results"):
    """
    Save Optuna study results to CSV files.
    
    Args:
        study: Optuna study object
        model_name: Name of model
        output_dir: Directory to save results
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best parameters
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df.insert(0, 'model', model_name)
    best_params_df.insert(1, 'best_nse', study.best_value)
    
    # Add best trial user attributes
    for key, value in study.best_trial.user_attrs.items():
        best_params_df[f'best_{key}'] = value
    
    best_params_path = os.path.join(output_dir, f"optuna_best_params_{model_name}.csv")
    best_params_df.to_csv(best_params_path, index=False)
    print(f"Best parameters saved to: {best_params_path}")
    
    # Save all trials
    trials_df = study.trials_dataframe()
    trials_path = os.path.join(output_dir, f"optuna_trials_{model_name}.csv")
    trials_df.to_csv(trials_path, index=False)
    print(f"All trials saved to: {trials_path}")
    
    return best_params_df, trials_df


# ============================================================================
# Sequential TCN/TCAN Objective Functions
# ============================================================================

def create_sequential_tcn_objective(train_dataset, val_dataset, device, max_epochs=400):
    """
    Create Optuna objective for Sequential TCN hyperparameter tuning.
    
    Args:
        train_dataset: SequentialIDFDataset for training
        val_dataset: SequentialIDFDataset for validation
        device: torch device
        max_epochs: Maximum epochs per trial
    
    Returns:
        Optuna objective function
    """
    def objective(trial):
        set_seed(SEED)
        
        # Suggest hyperparameters
        seq_len = trial.suggest_int("seq_len", 50, 500, step=50)
        extreme_percentile = trial.suggest_float("extreme_percentile", 70, 90)
        extreme_ratio = trial.suggest_float("extreme_ratio", 0.7, 0.95)
        num_filters = trial.suggest_categorical("num_filters", [64, 96, 128, 160, 192, 224, 256])
        kernel_size = trial.suggest_int("kernel_size", 3, 7)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        num_levels = trial.suggest_int("num_levels", 3, 6)
        lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        
        # Recreate datasets with suggested hyperparameters
        # Note: We'll need to pass these through or create datasets in the training script
        # For now, use the provided datasets
        
        # Create model
        model = SequentialTCN(
            num_channels=13,
            seq_len=train_dataset.seq_len,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout,
            num_levels=num_levels
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-7
        )
        
        # Create data loaders
        train_loader_gen = torch.Generator()
        train_loader_gen.manual_seed(SEED)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, generator=train_loader_gen
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop with early stopping (TCN)
        best_val_nse = -np.inf
        patience_counter = 0
        patience = 40  # Reduced for faster trials
        
        for epoch in range(max_epochs):
            train_loss = train_sequential_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate_sequential_model(model, val_loader, device, 
                                                    scaler_y=train_dataset.scaler_y)
            val_nse = val_metrics['nse']
            
            scheduler.step(val_nse)
            
            if val_nse > best_val_nse:
                best_val_nse = val_nse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Report to Optuna for pruning
            trial.report(val_nse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_nse
    
    return objective


def create_sequential_tcan_objective(train_dataset, val_dataset, device, max_epochs=400):
    """
    Create Optuna objective for Sequential TCAN hyperparameter tuning.
    
    Args:
        train_dataset: SequentialIDFDataset for training
        val_dataset: SequentialIDFDataset for validation
        device: torch device
        max_epochs: Maximum epochs per trial
    
    Returns:
        Optuna objective function
    """
    def objective(trial):
        set_seed(SEED)
        
        # Suggest hyperparameters
        seq_len = trial.suggest_int("seq_len", 50, 500, step=50)
        extreme_percentile = trial.suggest_float("extreme_percentile", 70, 90)
        extreme_ratio = trial.suggest_float("extreme_ratio", 0.7, 0.95)
        attn_heads = trial.suggest_categorical("attn_heads", [4, 8, 16])
        
        # num_filters must be divisible by attn_heads
        all_filters = [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384]
        valid_filters = [f for f in all_filters if f % attn_heads == 0]
        num_filters = trial.suggest_categorical("num_filters", valid_filters)
        
        kernel_size = trial.suggest_int("kernel_size", 3, 7)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        attn_dropout = trial.suggest_float("attn_dropout", 0.05, 0.2)
        num_levels = trial.suggest_int("num_levels", 3, 6)
        lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        
        # Create model
        model = SequentialTCAN(
            num_channels=13,
            seq_len=train_dataset.seq_len,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout,
            num_levels=num_levels,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-7
        )
        
        # Create data loaders
        train_loader_gen = torch.Generator()
        train_loader_gen.manual_seed(SEED)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, generator=train_loader_gen
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop with early stopping (TCAN)
        best_val_nse = -np.inf
        patience_counter = 0
        patience = 40  # Reduced for faster trials
        
        for epoch in range(max_epochs):
            train_loss = train_sequential_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate_sequential_model(model, val_loader, device, 
                                                    scaler_y=train_dataset.scaler_y)
            val_nse = val_metrics['nse']
            
            scheduler.step(val_nse)
            
            if val_nse > best_val_nse:
                best_val_nse = val_nse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Report to Optuna for pruning
            trial.report(val_nse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_nse
    
    return objective

