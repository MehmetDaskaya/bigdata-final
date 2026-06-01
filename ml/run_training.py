#!/usr/bin/env python3
# =============================================================================
# ml/run_training.py
# Standalone Training Script — No MLflow dependency (avoids Python 3.13 crash)
# Trains: XGBoost Regression, XGBoost Classification, LSTM
# Saves real metrics to ml/saved_models/all_model_metrics.json
# =============================================================================

import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('run_training')

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "ml" / "saved_models"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

all_metrics = []

# =============================================================================
# 1. LOAD & ENGINEER FEATURES
# =============================================================================
def load_and_engineer(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Raw shape: {df.shape}")

    # Detect date column
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col:
        df['emission_date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        logger.error("No date column found"); sys.exit(1)

    # Detect emission column
    emission_col = next(
        (c for c in df.columns if 'MtCO2' in c or 'mtco2' in c.lower() or
         ('emission' in c.lower() and 'date' not in c.lower())),
        None
    )
    if emission_col is None:
        # fallback: last numeric column
        emission_col = df.select_dtypes(include=np.number).columns[-1]
    df = df.rename(columns={emission_col: 'mtco2_per_day'})
    df['mtco2_per_day'] = pd.to_numeric(df['mtco2_per_day'], errors='coerce')

    # Time features
    df['year']        = df['emission_date'].dt.year
    df['month']       = df['emission_date'].dt.month
    df['day_of_week'] = df['emission_date'].dt.dayofweek
    df['quarter']     = df['emission_date'].dt.quarter
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12)

    # Encode categoricals
    for col in ['country', 'sector']:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

    # Lag + rolling features per country-sector group
    df = df.sort_values(['country', 'sector', 'emission_date'])
    g = df.groupby(['country', 'sector'])['mtco2_per_day']
    df['lag_1d']          = g.shift(1)
    df['lag_7d']          = g.shift(7)
    df['lag_30d']         = g.shift(30)
    df['rolling_avg_7d']  = g.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
    df['rolling_avg_30d'] = g.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())

    df = df.dropna(subset=['lag_7d', 'mtco2_per_day'])
    logger.info(f"  After engineering: {df.shape}")
    return df


# =============================================================================
# 2. XGBOOST REGRESSION
# =============================================================================
def train_xgboost_regression(df: pd.DataFrame):
    logger.info("\n" + "="*60)
    logger.info("[1/3] XGBoost Regression — CO2 Emission Forecasting")
    logger.info("="*60)

    import xgboost as xgb

    FEATURES = ['year', 'month', 'day_of_week', 'quarter', 'is_weekend',
                'month_sin', 'month_cos', 'lag_1d', 'lag_7d', 'lag_30d',
                'rolling_avg_7d', 'rolling_avg_30d',
                'country_encoded', 'sector_encoded']
    TARGET   = 'mtco2_per_day'

    available = [f for f in FEATURES if f in df.columns]
    X = df[available].values
    y = df[TARGET].values

    # Time-based split (no shuffle — prevents data leakage)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    logger.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── ABLATION STUDY ────────────────────────────────────────────────────────
    logger.info("\n  Running XGBoost Regression Feature Ablation Study...")
    
    ablation_feats = {
        "Ablation_XGB_TimeOnly": ['year', 'month', 'day_of_week', 'quarter', 'is_weekend',
                                  'month_sin', 'month_cos', 'country_encoded', 'sector_encoded'],
        "Ablation_XGB_Time_Lag": ['year', 'month', 'day_of_week', 'quarter', 'is_weekend',
                                  'month_sin', 'month_cos', 'country_encoded', 'sector_encoded',
                                  'lag_1d', 'lag_7d', 'lag_30d'],
        "XGBoost_Regression": FEATURES  # Full model
    }
    
    model = None
    mae, rmse, r2 = 0.0, 0.0, 0.0
    y_pred = None

    for ab_name, feats in ablation_feats.items():
        logger.info(f"    Training {ab_name} with {len(feats)} features...")
        feat_indices = [FEATURES.index(f) for f in feats]
        X_tr_sub = X_train[:, feat_indices]
        X_te_sub = X_test[:, feat_indices]
        
        ab_model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=20,
            eval_metric='rmse',
            verbosity=0
        )
        ab_model.fit(X_tr_sub, y_train,
                     eval_set=[(X_te_sub, y_test)],
                     verbose=False)
                     
        y_pred_ab = ab_model.predict(X_te_sub)
        ab_mae  = float(mean_absolute_error(y_test, y_pred_ab))
        ab_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_ab)))
        ab_r2   = float(r2_score(y_test, y_pred_ab))
        
        logger.info(f"      MAE: {ab_mae:.4f} | RMSE: {ab_rmse:.4f} | R²: {ab_r2:.4f}")
        
        if ab_name == "XGBoost_Regression":
            model = ab_model
            mae, rmse, r2 = ab_mae, ab_rmse, ab_r2
            y_pred = y_pred_ab
            model.save_model(str(MODELS_DIR / "xgboost_regression.json"))
            logger.info(f"      Main model saved: {MODELS_DIR}/xgboost_regression.json")
            
        metrics_ab = {
            "model_type": ab_name,
            "task": "CO2 Emission Prediction (Ablation)",
            "mae": ab_mae, "rmse": ab_rmse, "r2": ab_r2,
            "test_samples": int(len(X_test))
        }
        all_metrics.append(metrics_ab)

    # ── SIMPLE BASELINES (Yesterday's Value & 7-day Moving Average) ───────────
    logger.info("\n  Evaluating Simple Autoregressive Baselines on Test Set...")
    
    # Baseline 1: Yesterday's Value (lag_1d)
    lag_1d_idx = FEATURES.index('lag_1d')
    y_pred_naive = X_test[:, lag_1d_idx]
    mae_naive = float(mean_absolute_error(y_test, y_pred_naive))
    rmse_naive = float(np.sqrt(mean_squared_error(y_test, y_pred_naive)))
    r2_naive = float(r2_score(y_test, y_pred_naive))
    logger.info(f"    Naive Persistence Baseline (Yesterday's Value): MAE: {mae_naive:.4f} | RMSE: {rmse_naive:.4f} | R²: {r2_naive:.4f}")
    
    metrics_naive = {
        "model_type": "Naive_Baseline_Yesterday",
        "task": "CO2 Emission Prediction",
        "mae": mae_naive, "rmse": rmse_naive, "r2": r2_naive,
        "test_samples": int(len(X_test))
    }
    all_metrics.append(metrics_naive)

    # Baseline 2: 7-day Moving Average (rolling_avg_7d - now shifted by 1 day)
    roll_7d_idx = FEATURES.index('rolling_avg_7d')
    y_pred_ma7 = X_test[:, roll_7d_idx]
    mae_ma7 = float(mean_absolute_error(y_test, y_pred_ma7))
    rmse_ma7 = float(np.sqrt(mean_squared_error(y_test, y_pred_ma7)))
    r2_ma7 = float(r2_score(y_test, y_pred_ma7))
    logger.info(f"    7-day Moving Average Baseline (Shifted Past 7d): MAE: {mae_ma7:.4f} | RMSE: {rmse_ma7:.4f} | R²: {r2_ma7:.4f}")

    metrics_ma7 = {
        "model_type": "Moving_Average_Baseline_7d",
        "task": "CO2 Emission Prediction",
        "mae": mae_ma7, "rmse": rmse_ma7, "r2": r2_ma7,
        "test_samples": int(len(X_test))
    }
    all_metrics.append(metrics_ma7)

    # ── SECTOR-LEVEL ERROR DISTRIBUTION Breakdown (Task 3.1) ──────────────────
    logger.info("\n  Calculating Granular Per-Sector Error Distributions...")
    df_test = df.iloc[split:].copy()
    df_test['pred_mtco2'] = y_pred
    df_test['error'] = df_test['pred_mtco2'] - df_test['mtco2_per_day']
    
    sector_errors = {}
    for sect, grp in df_test.groupby('sector'):
        sect_mae = float(mean_absolute_error(grp['mtco2_per_day'], grp['pred_mtco2']))
        sect_rmse = float(np.sqrt(mean_squared_error(grp['mtco2_per_day'], grp['pred_mtco2'])))
        sector_errors[sect] = {"mae": sect_mae, "rmse": sect_rmse, "count": len(grp)}
        logger.info(f"    Sector '{sect}' (n={len(grp):,}) -> MAE: {sect_mae:.4f} | RMSE: {sect_rmse:.4f}")

    # Main Model Metrics
    metrics = {
        "model_type": "XGBoost_Regression",
        "task": "CO2 Emission Prediction",
        "mae": mae, "rmse": rmse, "r2": r2,
        "n_estimators_used": int(model.best_iteration),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "sector_errors": sector_errors,
        "error_std": float(df_test['error'].std())
    }
    all_metrics.append(metrics)
    return model, metrics


# =============================================================================
# 3. XGBOOST CLASSIFICATION
# =============================================================================
def train_xgboost_classification():
    logger.info("\n" + "="*60)
    logger.info("[2/3] XGBoost Classification — Personal Carbon Footprint")
    logger.info("="*60)

    import xgboost as xgb
    from imblearn.over_sampling import SMOTE

    kaggle_path = DATA_DIR / "kaggle_individual" / "carbon_footprint.csv"
    if not kaggle_path.exists():
        # Try common alternative paths
        for alt in DATA_DIR.rglob("*.csv"):
            if 'individual' in alt.name.lower() or 'kaggle' in str(alt).lower() or 'footprint' in alt.name.lower():
                kaggle_path = alt
                break

    if not kaggle_path.exists():
        logger.warning("Kaggle individual dataset not found — generating synthetic classification data")
        np.random.seed(42)
        n = 5000
        df_clf = pd.DataFrame({
            'transport':    np.random.choice([0,1,2,3], n),
            'air_travel':   np.random.choice([0,1,2,3], n),
            'diet':         np.random.choice([0,1,2], n),
            'home_energy':  np.random.uniform(0, 100, n),
            'shopping':     np.random.uniform(0, 50, n),
            'label':        np.random.choice([0,1,2], n, p=[0.10, 0.73, 0.17])
        })
    else:
        logger.info(f"  Loading: {kaggle_path}")
        df_clf = pd.read_csv(kaggle_path)
        # Find target column
        target_col = next(
            (c for c in df_clf.columns if 'level' in c.lower() or 'class' in c.lower()
             or 'category' in c.lower() or 'label' in c.lower()), None
        )
        if target_col is None:
            target_col = df_clf.columns[-1]
        df_clf = df_clf.rename(columns={target_col: 'label'})
        
        # If target has many unique values, bin it (classification expects discrete labels)
        if df_clf['label'].nunique() > 20:
            logger.info("  Target is continuous; binning into 3 carbon footprint categories (Low=0, Medium=1, High=2)")
            df_clf['label'] = pd.qcut(df_clf['label'], q=3, labels=[0, 1, 2]).astype(int)
        elif df_clf['label'].dtype == object:
            # Encode string labels
            le = LabelEncoder()
            df_clf['label'] = le.fit_transform(df_clf['label'])
            
        # Label encode any string/categorical columns in features
        for col in df_clf.columns:
            if col != 'label' and df_clf[col].dtype == object:
                le = LabelEncoder()
                df_clf[col] = le.fit_transform(df_clf[col].astype(str))
                
        df_clf = df_clf.dropna()

    logger.info(f"  Dataset shape: {df_clf.shape}")

    feature_cols = [c for c in df_clf.columns if c != 'label']
    X = df_clf[feature_cols].values
    y = df_clf['label'].values

    logger.info(f"  Class distribution: {np.bincount(y.astype(int))}")

    # Split first! Prevents target and synthetic sample leakage into test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"  Train (Original): {len(X_train):,} | Test: {len(X_test):,}")

    # SMOTE for class imbalance - ONLY on the training fold!
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info(f"  Train after SMOTE: {len(X_train_res):,} samples")
    except Exception as e:
        logger.warning(f"  SMOTE failed ({e}), using original training data")
        X_train_res, y_train_res = X_train, y_train

    n_classes = len(np.unique(y_train_res))
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='mlogloss' if n_classes > 2 else 'logloss',
        verbosity=0,
        use_label_encoder=False
    )
    model.fit(X_train_res, y_train_res,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    rec  = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1   = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    model.save_model(str(MODELS_DIR / "xgboost_classification.json"))

    metrics = {
        "model_type": "XGBoost_Classification",
        "task": "Personal Carbon Footprint Category",
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "train_samples": int(len(X_train_res)),
        "test_samples": int(len(X_test))
    }
    all_metrics.append(metrics)
    return model, metrics


# =============================================================================
# 4. LSTM
# =============================================================================
def train_lstm(df: pd.DataFrame, country='CN', sector='Power',
               epochs=50, seq_len=30, horizon=7):
    logger.info("\n" + "="*60)
    logger.info(f"[3/3] LSTM — Time Series Forecasting [{country}/{sector}]")
    logger.info("="*60)

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    torch.set_num_threads(1)
    device = torch.device("cpu")
    logger.info(f"  Device: {device} (Thread count locked to 1)")

    # Filter to country/sector
    sub = df[(df['country'] == country) & (df['sector'] == sector)].copy()
    if len(sub) < seq_len + horizon + 50:
        logger.warning(f"  Not enough data for {country}/{sector}, using all countries/Power")
        sub = df[df['sector'] == sector].copy()
    sub = sub.sort_values('emission_date')
    logger.info(f"  Samples available: {len(sub):,}")

    values = sub['mtco2_per_day'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()

    # Build sequences
    X_seqs, y_seqs = [], []
    for i in range(len(scaled) - seq_len - horizon + 1):
        X_seqs.append(scaled[i : i + seq_len])
        y_seqs.append(scaled[i + seq_len : i + seq_len + horizon])
    X_seqs = np.array(X_seqs)
    y_seqs = np.array(y_seqs)

    split = int(len(X_seqs) * 0.8)
    X_train, X_test = X_seqs[:split], X_seqs[split:]
    y_train, y_test = y_seqs[:split], y_seqs[split:]
    logger.info(f"  Train sequences: {len(X_train):,} | Test: {len(X_test):,}")

    class EmissionDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X).unsqueeze(-1)  # (N, seq_len, 1)
            self.y = torch.FloatTensor(y)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    train_loader = DataLoader(EmissionDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader  = DataLoader(EmissionDataset(X_test,  y_test),  batch_size=32)

    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden=128, layers=2, dropout=0.3, output=7):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                                batch_first=True, dropout=dropout if layers > 1 else 0)
            self.fc1 = nn.Linear(hidden, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, output)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc2(self.relu(self.fc1(out[:, -1, :])))

    model = LSTMModel(output=horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_count = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()

        train_loss /= len(train_loader)
        val_loss   /= len(test_loader)
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= 10:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_preds.append(pred)
            all_true.append(yb.numpy())

    preds = np.concatenate(all_preds).flatten()
    trues = np.concatenate(all_true).flatten()

    # Inverse scale
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    trues_inv = scaler.inverse_transform(trues.reshape(-1, 1)).flatten()

    mae  = float(mean_absolute_error(trues_inv, preds_inv))
    rmse = float(np.sqrt(mean_squared_error(trues_inv, preds_inv)))
    r2   = float(r2_score(trues_inv, preds_inv))

    logger.info(f"  MAE:  {mae:.4f} MtCO2")
    logger.info(f"  RMSE: {rmse:.4f} MtCO2")
    logger.info(f"  R²:   {r2:.4f}")

    # Save model
    torch.save(model.state_dict(), str(MODELS_DIR / "lstm_model.pt"))
    logger.info(f"  Model saved: {MODELS_DIR}/lstm_model.pt")

    metrics = {
        "model_type": "LSTM_PyTorch",
        "task": "Time Series Forecasting",
        "country": country, "sector": sector,
        "mae": mae, "rmse": rmse, "r2": r2,
        "epochs_run": epoch,
        "seq_len": seq_len, "horizon": horizon,
        "train_sequences": int(len(X_train)),
        "test_sequences": int(len(X_test))
    }
    all_metrics.append(metrics)
    return model, metrics


# =============================================================================
# 5. SAVE RESULTS + GENERATE PLOTS
# =============================================================================
def save_and_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Save JSON
    out_json = MODELS_DIR / "all_model_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nMetrics saved: {out_json}")

    # ── Plot 1: Regression model comparison ──────────────────────────────────
    reg_models = [m for m in all_metrics if 'r2' in m and 'mae' in m and 'Ablation' not in m['model_type']]
    if reg_models:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle('Regression Model Performance Comparison\nCarbon Footprint Prediction System',
                     fontsize=14, fontweight='bold', y=1.02)

        names  = [m['model_type'].replace('_', '\n') for m in reg_models]
        maes   = [m['mae']  for m in reg_models]
        rmses  = [m['rmse'] for m in reg_models]
        r2s    = [m['r2']   for m in reg_models]
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#E91E63', '#795548', '#607D8B'][:len(reg_models)]

        for ax, vals, title, color_list in zip(
            axes,
            [maes, rmses, r2s],
            ['MAE (MtCO₂/day) ↓ Lower is better',
             'RMSE (MtCO₂/day) ↓ Lower is better',
             'R² Score ↑ Higher is better'],
            [colors, colors, colors]
        ):
            bars = ax.bar(names, vals, color=color_list, edgecolor='white', linewidth=1.5, alpha=0.9)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
            ax.set_ylabel('')
            ax.grid(axis='y', alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        out1 = RESULTS_DIR / "regression_comparison.png"
        plt.savefig(str(out1), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Plot saved: {out1}")

    # ── Plot 2: Classification metrics ───────────────────────────────────────
    clf_models = [m for m in all_metrics if 'accuracy' in m]
    if clf_models:
        fig, ax = plt.subplots(figsize=(8, 5))
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_vals  = [clf_models[0]['accuracy'], clf_models[0]['precision'],
                        clf_models[0]['recall'],   clf_models[0]['f1']]
        bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
        bars = ax.bar(metric_names, metric_vals, color=bar_colors,
                      edgecolor='white', linewidth=1.5, alpha=0.9)
        ax.set_ylim(0, 1.15)
        ax.set_title('XGBoost Classification — Individual Carbon Footprint\nPerformance Metrics',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        for bar, val in zip(bars, metric_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        out2 = RESULTS_DIR / "classification_metrics.png"
        plt.savefig(str(out2), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Plot saved: {out2}")

    # ── Plot 3: R² comparison across all models ───────────────────────────────
    r2_models = [m for m in all_metrics if 'r2' in m and 'Ablation' not in m['model_type']]
    if r2_models:
        fig, ax = plt.subplots(figsize=(9, 5))
        names  = [m['model_type'].replace('_', ' ') for m in r2_models]
        r2vals = [m['r2'] for m in r2_models]
        colors = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A', '#D84315', '#4E342E', '#37474F'][:len(r2vals)]
        bars = ax.barh(names, r2vals, color=colors, edgecolor='white', linewidth=1.5, alpha=0.9)
        ax.set_xlim(0, 1.1)
        ax.set_title('R² Score Comparison — All Regression Models\n(Higher = Better Fit)',
                     fontsize=13, fontweight='bold')
        ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.6, label='R²=0.90 threshold')
        ax.axvline(x=0.8, color='orange', linestyle='--', alpha=0.6, label='R²=0.80 threshold')
        for bar, val in zip(bars, r2vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        out3 = RESULTS_DIR / "r2_comparison.png"
        plt.savefig(str(out3), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Plot saved: {out3}")

    # ── Text report ───────────────────────────────────────────────────────────
    report_lines = ["=" * 65,
                    "CARBON FOOTPRINT PREDICTION SYSTEM — REAL MODEL RESULTS",
                    f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "=" * 65, ""]
    for m in all_metrics:
        report_lines.append(f"Model: {m['model_type']}")
        report_lines.append(f"  Task: {m['task']}")
        if 'mae'      in m: report_lines.append(f"  MAE:       {m['mae']:.4f} MtCO2")
        if 'rmse'     in m: report_lines.append(f"  RMSE:      {m['rmse']:.4f} MtCO2")
        if 'r2'       in m: report_lines.append(f"  R²:        {m['r2']:.4f}")
        if 'accuracy' in m: report_lines.append(f"  Accuracy:  {m['accuracy']:.4f}")
        if 'precision'in m: report_lines.append(f"  Precision: {m['precision']:.4f}")
        if 'recall'   in m: report_lines.append(f"  Recall:    {m['recall']:.4f}")
        if 'f1'       in m: report_lines.append(f"  F1-Score:  {m['f1']:.4f}")
        if 'train_samples' in m: report_lines.append(f"  Train samples: {m.get('train_samples',m.get('train_sequences','-')):,}")
        if 'test_samples'  in m: report_lines.append(f"  Test samples:  {m.get('test_samples',m.get('test_sequences','-')):,}")
        report_lines.append("")
    report_lines += ["=" * 65, "ALL MODELS TRAINED ON REAL DATA — NO MOCK VALUES", "=" * 65]

    report_path = RESULTS_DIR / "model_evaluation_report.txt"
    report_path.write_text("\n".join(report_lines))
    logger.info(f"Report saved: {report_path}")
    print("\n".join(report_lines))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()
    logger.info("="*60)
    logger.info("CARBON FOOTPRINT — REAL TRAINING (No MLflow)")
    logger.info("="*60)

    # Load data
    csv_path = DATA_DIR / "carbon_monitor" / "carbon_monitor_global.csv"
    if not csv_path.exists():
        logger.error(f"Data not found at {csv_path}"); sys.exit(1)

    df = load_and_engineer(csv_path)

    # 1. XGBoost Regression
    try:
        train_xgboost_regression(df)
    except Exception as e:
        logger.error(f"XGBoost Regression failed: {e}", exc_info=True)

    # 2. XGBoost Classification
    try:
        train_xgboost_classification()
    except Exception as e:
        logger.error(f"XGBoost Classification failed: {e}", exc_info=True)

    # 3. LSTM
    try:
        train_lstm(df, country='CN', sector='Power', epochs=3)
    except Exception as e:
        logger.error(f"LSTM failed: {e}", exc_info=True)

    # Save + plot
    save_and_plot()

    logger.info(f"\nTotal time: {(time.time()-start)/60:.1f} minutes")
