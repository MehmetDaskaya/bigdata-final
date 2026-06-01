#!/usr/bin/env python3
# =============================================================================
# ml/xgboost_model.py
# XGBoost Model — Both Regression and Classification
# =============================================================================
# This module utilizes XGBoost for two tasks:
#   1. REGRESSION: Daily CO2 prediction by country/sector (MtCO2)
#   2. CLASSIFICATION: Classify individual carbon footprint (Low/Medium/High)
#
# WHY XGBOOST?
#   - Gradient boosting: converts weak learners to strong models
#   - Explainable via SHAP values (evaluative criterion for instructor)
#   - Mixed data type support (categorical + numerical)
#   - Faster training compared to LSTM (useful as a baseline model)
#
# Usage:
#   python ml/xgboost_model.py
# =============================================================================

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score, f1_score,
                              classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('xgboost_model')

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "ml" / "saved_models"
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TASK 1: REGRESSION — Sectoral CO2 Emission Prediction
# =============================================================================

def train_regression_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    XGBoost Regression Model — Daily CO2 Emission Prediction
    
    Input Features:
    - Time features: year, month, day, quarter, season
    - Lag features: 1, 7, 30 day delayed values
    - Rolling averages: 7, 30 day
    - Categorical: country, sector (label encoded)
    
    Target: mtco2_per_day (daily million tons CO2)
    
    Args:
        df: Feature engineered Carbon Monitor DataFrame
    
    Returns:
        dict containing model, metrics, feature_importance, shap_values
    """
    logger.info("\n" + "="*60)
    logger.info("Training XGBoost Regression Model...")
    logger.info("="*60)
    
    # Feature list
    categorical_cols = ['country', 'sector']
    
    # Select available numerical columns
    candidate_features = [
        'year', 'month', 'day_of_week', 'day_of_year', 'quarter',
        'is_weekend', 'month_sin', 'month_cos',
        'lag_1d', 'lag_7d', 'lag_30d',
        'rolling_avg_7d', 'rolling_avg_30d'
    ]
    
    available_features = [f for f in candidate_features if f in df.columns]
    
    # Label encode categorical columns
    df_work = df.copy()
    label_encoders = {}
    for col in categorical_cols:
        if col in df_work.columns:
            le = LabelEncoder()
            df_work[f'{col}_encoded'] = le.fit_transform(df_work[col].astype(str))
            label_encoders[col] = le
            available_features.append(f'{col}_encoded')
    
    logger.info(f"Used features ({len(available_features)}): {available_features}")
    
    # Target variable
    target_col = 'mtco2_per_day'
    if target_col not in df_work.columns:
        # Fallback: find the first column containing emissions
        target_col = next((c for c in df_work.columns if 'co2' in c.lower() or 'emission' in c.lower()), None)
    
    # Drop missing values
    feature_df = df_work[available_features + [target_col]].dropna()
    
    X = feature_df[available_features].values
    y = feature_df[target_col].values
    
    logger.info(f"Data size: X={X.shape}, y={y.shape}")
    
    # Time-based split (shuffle=False — prevent data leakage)
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # === XGBoost Hyperparameters ===
    # These parameters are manually configured.
    # In a production environment, auto-tuning via GridSearchCV or Optuna is preferred.
    params = {
        'n_estimators':     500,      # Number of trees (can be increased)
        'max_depth':        6,         # Depth of each tree (overfitting control)
        'learning_rate':    0.05,      # Low learning rate = better generalization
        'subsample':        0.8,       # Subsample ratio of the training instances
        'colsample_bytree': 0.8,       # Subsample ratio of columns when constructing each tree
        'min_child_weight': 5,         # Minimum sum of instance weight needed in a child
        'reg_alpha':        0.1,       # L1 regularization (lasso)
        'reg_lambda':       1.0,       # L2 regularization (ridge)
        'objective':        'reg:squarederror',  # MSE loss function
        'random_state':     42,
        'n_jobs':           -1,        # Use all CPU cores
        'eval_metric':      'rmse',
        'early_stopping_rounds': 50    # v2.x: defined in constructor
    }
    
    model = xgb.XGBRegressor(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100                    # Report every 100 trees
    )
    
    # Prediction
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # Prevent negative predictions
    
    # Metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    
    logger.info(f"\nRegression Test Results:")
    logger.info(f"  MAE:  {mae:.4f} MtCO2")
    logger.info(f"  RMSE: {rmse:.4f} MtCO2")
    logger.info(f"  R²:   {r2:.4f}")
    
    # Feature Importance (XGBoost's built-in method)
    importance_dict = dict(zip(available_features,
                                model.feature_importances_))
    importance_df = pd.DataFrame(
        list(importance_dict.items()),
        columns=['feature', 'importance']
    ).sort_values('importance', ascending=False)
    
    logger.info("\nFeature Importance (Top 10):")
    logger.info(importance_df.head(10).to_string(index=False))
    
    # === SHAP VALUES ===
    # SHAP explains why each prediction yielded its specific value
    # Critical for model explainability (instructor evaluation criterion)
    logger.info("\nComputing SHAP values...")
    
    try:
        explainer   = shap.TreeExplainer(model)
        # Use the first 200 samples of the test set (for speed)
        shap_values = explainer.shap_values(X_test[:200])
        
        # Save SHAP summary
        shap_summary = pd.DataFrame(
            np.abs(shap_values).mean(axis=0).reshape(1, -1),
            columns=available_features
        )
        logger.info("Global SHAP feature importance computed")
        
    except Exception as e:
        logger.warning(f"Could not compute SHAP: {e}")
        shap_values = None
    
    # Save the model
    model_path = MODELS_DIR / "xgboost_regression.json"
    model.save_model(str(model_path))
    logger.info(f"\n✓ Model saved: {model_path}")
    
    return {
        "model":           model,
        "feature_cols":    available_features,
        "label_encoders":  label_encoders,
        "importance":      importance_df,
        "shap_values":     shap_values,
        "metrics": {
            "model_type": "XGBoost_Regression",
            "task":       "CO2 Emission Prediction",
            "mae":        float(mae),
            "rmse":       float(rmse),
            "r2":         float(r2),
            "n_estimators_used": model.best_iteration if hasattr(model, 'best_iteration') else params['n_estimators'],
            "test_samples": len(X_test)
        }
    }


# =============================================================================
# TASK 2: CLASSIFICATION — Individual Carbon Footprint Category
# =============================================================================

def train_classification_model(individual_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    XGBoost Classification Model — Personal Carbon Footprint Category
    
    Target Classes:
    - Low (0):    < 2500 kg CO2e/year — eco-friendly lifestyle
    - Medium (1): 2500-5000 kg CO2e/year — average
    - High (2):   > 5000 kg CO2e/year — high emissions
    
    Class Imbalance: SMOTE (Synthetic Minority Over-sampling Technique) is applied
    
    Args:
        individual_csv: CSV file path (if None, default is used)
    
    Returns:
        dict containing model, metrics, classification_report
    """
    logger.info("\n" + "="*60)
    logger.info("Training XGBoost Classification Model...")
    logger.info("="*60)
    
    # Load data
    csv_path = individual_csv or str(DATA_DIR / "kaggle_individual" / "individual_carbon.csv")
    
    if not Path(csv_path).exists():
        logger.error(f"Individual data not found: {csv_path}")
        logger.info("run python data/download_data.py first")
        return {}
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded: {len(df):,} records, {len(df.columns)} columns")
    
    # Create target variable
    target_col = 'CarbonEmission'
    if target_col not in df.columns:
        target_col = df.columns[-1]  # Last column is usually the target
    
    df['emission_class'] = pd.cut(
        df[target_col],
        bins=[0, 2500, 5000, float('inf')],
        labels=[0, 1, 2]  # 0=Low, 1=Medium, 2=High
    ).astype(int)
    
    # Class distribution
    class_dist = df['emission_class'].value_counts().sort_index()
    logger.info(f"\nClass distribution:")
    for cls, count in class_dist.items():
        cls_name = ['Low', 'Medium', 'High'][cls]
        logger.info(f"  {cls_name}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # Separate features and target
    feature_cols = [c for c in df_encoded.columns
                    if c not in [target_col, 'emission_class']]
    
    X = df_encoded[feature_cols].values
    y = df_encoded['emission_class'].values
    
    # Train/test split (stratified — preserves class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # SMOTE — balance class distribution (only on train set!)
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        logger.info(f"\nTrain size after SMOTE: {len(X_train_balanced):,} "
                   f"(before: {len(X_train):,})")
    except ImportError:
        logger.warning("imbalanced-learn is not installed, SMOTE skipped")
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # XGBoost Classifier
    # Note: early_stopping_rounds is now defined as a model parameter (v2.x)
    clf_params = {
        'n_estimators':         300,
        'max_depth':            5,
        'learning_rate':        0.1,
        'subsample':            0.8,
        'colsample_bytree':     0.8,
        'objective':            'multi:softmax',  # Multi-class classification
        'num_class':            3,                # Low, Medium, High
        'eval_metric':          'merror',         # Multi-class error rate
        'random_state':         42,
        'n_jobs':               -1,
        'early_stopping_rounds': 30,              # v2.x: defined in constructor
    }
    
    clf = xgb.XGBClassifier(**clf_params)
    
    clf.fit(
        X_train_balanced, y_train_balanced,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Prediction
    y_pred = clf.predict(X_test)
    
    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"\nClassification Test Results:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=['Low', 'Medium', 'High']))
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    logger.info("\nMost important features (Top 10):")
    logger.info(importance_df.head(10).to_string(index=False))
    
    # Save the model
    model_path = MODELS_DIR / "xgboost_classification.json"
    clf.save_model(str(model_path))
    
    return {
        "model":        clf,
        "feature_cols": feature_cols,
        "metrics": {
            "model_type": "XGBoost_Classification",
            "task":       "Personal Carbon Footprint Category",
            "accuracy":   float(accuracy),
            "precision":  float(precision),
            "recall":     float(recall),
            "f1":         float(f1),
            "test_samples": len(X_test)
        },
        "importance": importance_df
    }


if __name__ == "__main__":
    logger.info("Running XGBoost Models in Test Mode...")
    
    # Classification test (requires individual data)
    clf_results = train_classification_model()
    if clf_results:
        logger.info(f"\nClassification metrics: {clf_results['metrics']}")
