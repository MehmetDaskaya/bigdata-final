#!/usr/bin/env python3
# =============================================================================
# ml/lstm_model.py
# LSTM Deep Learning Model — Time Series CO2 Forecasting
# =============================================================================
# This module defines and trains a Long Short-Term Memory (LSTM) network 
# that makes 7-day carbon emission predictions.
#
# WHY LSTM?
#   - Emission data is time-dependent (yesterday's emission affects today)
#   - LSTM can learn long-term dependencies (superior to vanilla RNN)
#   - Automatically learns patterns like seasonality and trends
#
# ARCHITECTURE:
#   Input: 30-day window (30, feature_count) → LSTM Layer →
#   Dropout → LSTM Layer → Fully Connected Layer → 7-day prediction
#
# Usage:
#   from ml.lstm_model import LSTMEmissionPredictor, train_lstm
#   predictor = train_lstm(df)
# =============================================================================

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('lstm_model')

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "ml" / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LSTM Model Architecture
# =============================================================================

class LSTMEmissionModel(nn.Module):
    """
    Multi-Layer LSTM Model — Multi-Step Time Series Forecasting
    
    Architecture Details:
    ┌────────────────────────────────────────────┐
    │ Input: (batch, seq_len=30, features=N)     │
    │ → LSTM 1 (hidden=128, num_layers=2)        │
    │ → Dropout (p=0.3)                          │
    │ → LSTM 2 (hidden=64)                       │
    │ → Fully Connected (64 → 32)                │
    │ → ReLU                                     │
    │ → Fully Connected (32 → output_steps=7)    │
    │ Output: (batch, 7) — 7-day prediction      │
    └────────────────────────────────────────────┘
    
    Args:
        input_size: Feature count
        hidden_size: LSTM hidden layer size
        num_layers: LSTM layer count (stacked LSTM)
        output_steps: Number of days to forecast (7)
        dropout: Dropout rate to prevent overfitting
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, output_steps: int = 7, dropout: float = 0.3):
        super(LSTMEmissionModel, self).__init__()
        
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.output_steps = output_steps
        
        # First LSTM layer — learns time series patterns
        # batch_first=True: accepts input in (batch, seq, feature) format
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # For multi-layer LSTM
        )
        
        # Dropout — overfitting prevention (randomly drop neurons during training)
        self.dropout = nn.Dropout(p=dropout)
        
        # Second LSTM layer — learns higher-level features
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # Dimensionality reduction
            num_layers=1,
            batch_first=True
        )
        
        # Fully connected classifier layers
        self.fc1 = nn.Linear(hidden_size // 2, 32)
        self.relu = nn.ReLU()               # Non-linear activation
        self.fc2 = nn.Linear(32, output_steps)  # 7-day prediction output
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass calculation.
        
        Args:
            x: input tensor shape (batch_size, seq_len, input_size)
        
        Returns:
            prediction tensor shape (batch_size, output_steps)
        """
        batch_size = x.size(0)
        
        # Initial hidden state of LSTM — initialize with zeros
        h0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Pass through first LSTM layer
        # out: (batch, seq_len, hidden_size) — output of each time step
        out1, _ = self.lstm1(x, (h0_1, c0_1))
        out1    = self.dropout(out1)
        
        # Pass through second LSTM layer
        h0_2 = torch.zeros(1, batch_size, self.hidden_size // 2).to(x.device)
        c0_2 = torch.zeros(1, batch_size, self.hidden_size // 2).to(x.device)
        
        out2, _ = self.lstm2(out1, (h0_2, c0_2))
        
        # Use only the last time step (all steps can be used for many-to-many)
        # [-1] = last step → (batch, hidden_size//2)
        last_output = out2[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.fc2(out)  # (batch, output_steps=7)
        
        return out


class EmissionTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset — Time Series Window Dataset
    
    Each sample:
    - X: 30-day emission and feature window (input)
    - y: Subsequent 7-day emission values (target)
    
    Sliding window approach:
    [d1,...,d30] → [d31,...,d37]
    [d2,...,d31] → [d32,...,d38]
    ...
    
    Args:
        features: Feature matrix (n_samples, n_features)
        targets: Target emission values (n_samples,)
        seq_len: Input window size (default: 30)
        horizon: Forecast horizon (default: 7)
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 seq_len: int = 30, horizon: int = 7):
        self.features = torch.FloatTensor(features)
        self.targets  = torch.FloatTensor(targets)
        self.seq_len  = seq_len
        self.horizon  = horizon
        
    def __len__(self) -> int:
        # Total sample count: all data - window - horizon + 1
        return len(self.features) - self.seq_len - self.horizon + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input window: [idx, idx+seq_len)
        x = self.features[idx : idx + self.seq_len]
        # Target window: [idx+seq_len, idx+seq_len+horizon)
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return x, y


def prepare_features(df: pd.DataFrame, country: str = "CN",
                     sector: str = "Power") -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Transforms raw DataFrame into a feature matrix for LSTM training.
    
    Used features:
    - mtco2_per_day: Primary target (also used as input)
    - rolling_avg_7d, rolling_avg_30d: Short/long term trend
    - lag_1d, lag_7d, lag_30d: Temporal dependency
    - month_sin, month_cos: Cyclical seasonality
    - is_weekend: Weekend effect
    
    Normalization: MinMaxScaler scales to [0, 1] range
    (LSTMs learn better with normalized data)
    
    Args:
        df: Feature engineered DataFrame
        country: Filter country
        sector: Filter sector
    
    Returns:
        (features, targets, scaler) tuple
    """
    # Filter by country and sector
    mask = (df['country'] == country) & (df['sector'] == sector)
    subset = df[mask].sort_values('date' if 'date' in df.columns else 'emission_date').copy()
    
    if len(subset) < 100:
        raise ValueError(f"insufficient data for {country}/{sector} ({len(subset)} records, min 100)")
    
    # Feature columns — all must be numerical
    feature_cols = [
        'mtco2_per_day',      # Primary target (also used as input)
        'rolling_avg_7d',     # 7-day average
        'rolling_avg_30d',    # 30-day average
        'lag_1d',             # Yesterday's emission
        'lag_7d',             # Last week's emission
        'lag_30d',            # Last month's emission
        'month_sin',          # Sinusoidal month encoding
        'month_cos',          # Cosine month encoding
        'is_weekend',         # Weekend flag
    ]
    
    # Filter from available columns
    available_cols = [c for c in feature_cols if c in subset.columns]
    logger.info(f"Used features: {available_cols}")
    
    subset_clean = subset[available_cols].dropna()
    
    # Normalize using MinMaxScaler
    scaler  = MinMaxScaler(feature_range=(0, 1))
    scaled  = scaler.fit_transform(subset_clean.values)
    
    features = scaled                       # All features (input)
    targets  = scaled[:, 0]                 # First column = mtco2_per_day (target)
    
    logger.info(f"Feature matrix: {features.shape} | Target: {targets.shape}")
    return features, targets, scaler


def train_lstm(df: pd.DataFrame, country: str = "CN", sector: str = "Power",
               epochs: int = 50, batch_size: int = 32,
               learning_rate: float = 0.001) -> dict:
    """
    Trains and evaluates the LSTM model.
    
    Training process:
    1. Split data into 80/10/10 train/val/test (chronologically)
    2. Apply mini-batch gradient descent at each epoch
    3. Monitor validation loss (for early stopping)
    4. Perform final evaluation on test set
    
    Args:
        df: Feature engineered DataFrame
        country: Target country
        sector: Target sector
        epochs: Training epoch count
        batch_size: Mini-batch size
        learning_rate: Adam optimizer learning rate
    
    Returns:
        Results dict (model, metrics, predictions)
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"LSTM Training: {country} / {sector}")
    logger.info(f"{'='*50}")
    
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Feature preparation
    features, targets, scaler = prepare_features(df, country, sector)
    
    # 80/10/10 time-based split (no shuffle! crucial for time series)
    n = len(features)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)
    
    train_features, train_targets = features[:train_end],  targets[:train_end]
    val_features,   val_targets   = features[train_end:val_end], targets[train_end:val_end]
    test_features,  test_targets  = features[val_end:],    targets[val_end:]
    
    logger.info(f"Train: {len(train_features)} | Val: {len(val_features)} | Test: {len(test_features)}")
    
    # Create Dataset and DataLoader
    SEQ_LEN = 30   # 30-day input window
    HORIZON  = 7   # 7-day forecast horizon
    
    train_ds = EmissionTimeSeriesDataset(train_features, train_targets, SEQ_LEN, HORIZON)
    val_ds   = EmissionTimeSeriesDataset(val_features,   val_targets,   SEQ_LEN, HORIZON)
    test_ds  = EmissionTimeSeriesDataset(test_features,  test_targets,  SEQ_LEN, HORIZON)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)  # shuffle=False!
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    # Create model
    input_size = features.shape[1]  # Feature count
    model = LSTMEmissionModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        output_steps=HORIZON,
        dropout=0.3
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()        # Mean Squared Error — standard for regression
    optimizer = torch.optim.Adam(   # Adam — adaptive learning rate
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5           # L2 regularization — prevent overfitting
    )
    # Learning rate scheduler — reduce learning rate (on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )
    
    # === TRAINING LOOP ===
    train_losses = []
    val_losses   = []
    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOP_PATIENCE = 10  # Stop if no improvement for 10 epochs
    
    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        epoch_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()         # Zero gradients
            predictions = model(batch_x)  # Forward pass
            loss = criterion(predictions, batch_y)  # Calculate loss
            loss.backward()               # Backward pass
            
            # Gradient clipping — prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()              # Update weights
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # --- Validation phase ---
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():  # Disable gradient calculation for memory efficiency and speed
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                epoch_val_loss += criterion(preds, batch_y).item()
        
        avg_val_loss = epoch_val_loss / max(len(val_loader), 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Report every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch [{epoch:3d}/{epochs}] | "
                        f"Train Loss: {avg_train_loss:.6f} | "
                        f"Val Loss: {avg_val_loss:.6f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       MODELS_DIR / f"lstm_best_{country}_{sector.replace(' ', '_')}.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
                break
    
    # === TEST EVALUATION ===
    # Load the best model
    model.load_state_dict(
        torch.load(MODELS_DIR / f"lstm_best_{country}_{sector.replace(' ', '_')}.pt",
                   map_location=device)
    )
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.numpy().tolist())
    
    # Convert to Numpy arrays
    preds_arr   = np.array(all_preds).flatten()
    targets_arr = np.array(all_targets).flatten()
    
    # Metrics calculation
    mae  = mean_absolute_error(targets_arr, preds_arr)
    rmse = np.sqrt(mean_squared_error(targets_arr, preds_arr))
    r2   = r2_score(targets_arr, preds_arr)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"LSTM Test Results — {country}/{sector}")
    logger.info(f"{'='*50}")
    logger.info(f"  MAE  (Mean Absolute Error):       {mae:.6f}")
    logger.info(f"  RMSE (Root Mean Squared Error):   {rmse:.6f}")
    logger.info(f"  R²   (R-squared Score):           {r2:.4f}")
    logger.info(f"{'='*50}")
    
    return {
        "model":        model,
        "scaler":       scaler,
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "metrics": {
            "model_type": "LSTM",
            "country":    country,
            "sector":     sector,
            "mae":        float(mae),
            "rmse":       float(rmse),
            "r2":         float(r2),
            "epochs_run": len(train_losses),
            "best_val_loss": float(best_val_loss)
        }
    }


if __name__ == "__main__":
    # Fast training with synthetic data for testing
    logger.info("Running LSTM Model in Test Mode...")
    
    # Create simple test data
    np.random.seed(42)
    n = 500
    test_df = pd.DataFrame({
        'date':           pd.date_range('2020-01-01', periods=n, freq='D'),
        'country':        'CN',
        'sector':         'Power',
        'mtco2_per_day':  2.5 + 0.3 * np.sin(np.linspace(0, 8*np.pi, n)) + np.random.normal(0, 0.05, n),
        'rolling_avg_7d': 2.5 + np.random.normal(0, 0.02, n),
        'rolling_avg_30d':2.5 + np.random.normal(0, 0.01, n),
        'lag_1d':         2.5 + np.random.normal(0, 0.05, n),
        'lag_7d':         2.5 + np.random.normal(0, 0.05, n),
        'lag_30d':        2.5 + np.random.normal(0, 0.05, n),
        'month_sin':      np.sin(2 * np.pi * np.arange(n) / 365),
        'month_cos':      np.cos(2 * np.pi * np.arange(n) / 365),
        'is_weekend':     (pd.date_range('2020-01-01', periods=n, freq='D').dayofweek >= 5).astype(int),
    })
    
    results = train_lstm(test_df, country="CN", sector="Power", epochs=20, batch_size=16)
    logger.info(f"\nTest successful! Metrics: {results['metrics']}")
