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

def create_svm_objective(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled):
    """
    Create Optuna objective function for SVM hyperparameter tuning.
    
    Args:
        X_train_scaled: Scaled training features (numpy array)
        y_train_scaled: Scaled training targets (numpy array)
        X_val_scaled: Scaled validation features (numpy array)
        y_val_scaled: Scaled validation targets (numpy array)
    
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
        
        # Predict on validation set
        y_pred_scaled = model.predict(X_val_scaled)
        
        # Evaluate (maximize NSE)
        metrics = evaluate_predictions(y_val_scaled, y_pred_scaled)
        
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
        scaler_y: Optional scaler to inverse transform predictions
    
    Returns:
        dict with metrics (nse, rmse, mae, r2)
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
    
    # Inverse transform if scaler provided
    if scaler_y is not None:
        preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
        trues = scaler_y.inverse_transform(trues.reshape(-1, 1)).flatten()
    
    return evaluate_predictions(trues, preds)


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
    """Temporal block for TCN with causal convolutions and residual connections."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.01):
        super().__init__()
        padding = (kernel_size - 1)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]] if self.conv2.padding[0] > 0 else out
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network for regression."""
    def __init__(self, seq_len=2, num_filters=144, kernel_size=3, dropout=0.01, num_levels=2):
        super().__init__()
        self.seq_len = seq_len
        
        layers = []
        in_channels = seq_len
        for _ in range(num_levels):
            layers.append(TemporalBlock(in_channels, num_filters, kernel_size, dropout))
            in_channels = num_filters
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters, 1)
    
    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, 1) for treating features as channels
        x = x.unsqueeze(-1)
        # Transpose to (batch, 1, seq_len) -> Wait, NO! We want (batch, seq_len, 1) to be (batch, in_channels=seq_len, seq_length=1)
        # Actually for Conv1d: input is (batch, channels, length)
        # We have seq_len features, so they should be channels
        # So: (batch, seq_len) -> add length dimension -> (batch, seq_len, 1)
        # This gives us (batch, channels=seq_len, length=1)
        x = self.network(x)
        x = x[:, :, -1]  # Take last time step (last position in sequence)
        return self.fc(x)


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
        
        # Suggest hyperparameters - AGGRESSIVE SEARCH SPACE FOR NSE > 0.95
        num_filters = trial.suggest_categorical("num_filters", [128, 192, 256, 384, 512, 640, 768])
        kernel_size = trial.suggest_int("kernel_size", 2, 9)  # Include smaller and larger
        dropout = trial.suggest_float("dropout", 0.0, 0.3)  # Allow NO dropout
        num_levels = trial.suggest_int("num_levels", 1, 5)  # Include shallow nets
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)  # Much wider range
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        
        # Create model
        model = TCN(seq_len=X_train_scaled.shape[1],
                    num_filters=num_filters,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    num_levels=num_levels).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-7
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
        patience = 100
        patience_counter = 0
        
        # Training loop
        for epoch in range(max_epochs):
            train_pytorch_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            
            # Evaluate on validation - WITH INVERSE TRANSFORM FOR PROPER NSE
            metrics = evaluate_pytorch_model(model, val_loader, device, scaler_y=scaler_y)
            val_nse = metrics['nse']
            
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
    """Multi-Head Self-Attention for 1D sequences."""
    def __init__(self, channels, num_heads=8, dropout=0.05):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        attn_out, _ = self.mha(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        # (B, L, C) -> (B, C, L)
        return x.transpose(1, 2)


class TemporalBlockWithAttention(nn.Module):
    """Temporal block with optional self-attention."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.01, 
                 attn_heads=0, attn_dropout=0.05):
        super().__init__()
        padding = (kernel_size - 1)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.attn = MHSA1d(out_channels, attn_heads, attn_dropout) if attn_heads > 0 else None
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]] if self.conv2.padding[0] > 0 else out
        out = self.relu2(out)
        out = self.dropout2(out)
        
        if self.attn is not None:
            out = self.attn(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCAN(nn.Module):
    """Temporal Convolutional Attention Network for regression."""
    def __init__(self, seq_len=2, num_filters=144, kernel_size=3, dropout=0.01, 
                 num_levels=2, attn_heads=8, attn_dropout=0.05):
        super().__init__()
        self.seq_len = seq_len
        
        layers = []
        in_channels = seq_len
        for _ in range(num_levels):
            layers.append(TemporalBlockWithAttention(
                in_channels, num_filters, kernel_size, dropout, attn_heads, attn_dropout
            ))
            in_channels = num_filters
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters, 1)
    
    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, 1) for treating features as channels
        # Conv1d expects (batch, channels, length), so seq_len features become channels
        x = x.unsqueeze(-1)
        x = self.network(x)
        x = x[:, :, -1]  # Take last time step (last position in sequence)
        return self.fc(x)


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
        
        # Suggest hyperparameters - AGGRESSIVE SEARCH SPACE FOR NSE > 0.96
        num_filters = trial.suggest_categorical("num_filters", [192, 256, 384, 512, 640, 768, 896])
        kernel_size = trial.suggest_int("kernel_size", 2, 9)  # Include smaller and larger
        dropout = trial.suggest_float("dropout", 0.0, 0.35)  # Allow NO dropout, higher ceiling
        num_levels = trial.suggest_int("num_levels", 1, 6)  # Include shallow and very deep
        attn_heads = trial.suggest_categorical("attn_heads", [2, 4, 6, 8, 12, 16])  # More variety
        attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.35)  # Allow NO dropout
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)  # Much wider range
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        
        # Create model
        model = TCAN(seq_len=X_train_scaled.shape[1],
                     num_filters=num_filters,
                     kernel_size=kernel_size,
                     dropout=dropout,
                     num_levels=num_levels,
                     attn_heads=attn_heads,
                     attn_dropout=attn_dropout).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-7
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
        patience = 100
        patience_counter = 0
        
        # Training loop
        for epoch in range(max_epochs):
            train_pytorch_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            
            # Evaluate on validation - WITH INVERSE TRANSFORM FOR PROPER NSE
            metrics = evaluate_pytorch_model(model, val_loader, device, scaler_y=scaler_y)
            val_nse = metrics['nse']
            
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
