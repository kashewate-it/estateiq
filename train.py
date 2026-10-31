"""
train.py
Machine Learning Component 1: House Price Regression Pipeline.
Trains, validates, compares, and serializes regression models for price prediction.
"""

import sys
import time
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from utils import (
    load_housing_data,
    build_preprocessor,
    save_metadata,
    DATA_PATH,
    MODELS_DIR,
    REGRESSION_MODEL_PATH,
    REGRESSION_META_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_INPUT_FEATURES,
    TARGET_COL,
)


def train_regression_models():
    """
    Train and evaluate regression models.
    Uses 5-fold Cross-Validation on the training split to select the champion model,
    then performs unbiased evaluation on the holdout test set.
    """
    print("=" * 75)
    print(" EstateIQ: Machine Learning Component 1 — Price Prediction Pipeline")
    print("=" * 75)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[1] Loading dataset from '{DATA_PATH}'...")
    df = load_housing_data(DATA_PATH)
    print(f"    Loaded {len(df):,} valid property records.")

    # Prevent target leakage: Drop id, date, price from feature matrix
    X = df[ALL_INPUT_FEATURES].copy()
    y = df[TARGET_COL].values.copy()

    print(f"\n[2] Features configured:")
    print(f"    - Numerical features ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
    print(f"    - Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    print(f"    - Target feature: '{TARGET_COL}' (excluded from X)")

    # 80/20 train/test split with fixed seed
    RANDOM_SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    print(f"\n[3] Train/Test Partitioning:")
    print(f"    - Training records : {len(X_train):,} (80%)")
    print(f"    - Test records     : {len(X_test):,} (20%)")

    # Candidate models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=RANDOM_SEED),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=80,
            max_depth=14,
            min_samples_split=4,
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
    }

    print(f"\n[4] Model Selection via 3-Fold Cross Validation on Training Split...")
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    cv_results = {}
    fitted_pipelines = {}

    for name, regressor in candidate_models.items():
        print(f"    --> Evaluating {name}...")
        preprocessor = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

        start_time = time.time()
        scoring = {
            "r2": "r2",
            "neg_mae": "neg_mean_absolute_error",
            "neg_rmse": "neg_root_mean_squared_error",
        }
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        fit_duration = time.time() - start_time

        mean_r2 = float(np.mean(cv_scores["test_r2"]))
        mean_mae = float(-np.mean(cv_scores["test_neg_mae"]))
        mean_rmse = float(-np.mean(cv_scores["test_neg_rmse"]))

        cv_results[name] = {
            "cv_r2": round(mean_r2, 4),
            "cv_mae": round(mean_mae, 2),
            "cv_rmse": round(mean_rmse, 2),
            "cv_time_sec": round(fit_duration, 2),
        }
        print(f"        CV R2: {mean_r2:.4f} | CV MAE: ${mean_mae:,.2f} | CV RMSE: ${mean_rmse:,.2f} ({fit_duration:.1f}s)")

        # Fit on full training set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

    # Select champion model based on CV R² score
    best_model_name = max(cv_results, key=lambda k: cv_results[k]["cv_r2"])
    print(f"\n[5] Model Selection Result: '{best_model_name}' selected based on CV performance.")

    # [6] Final Unbiased Evaluation on Test Set
    print(f"\n[6] Final Evaluation on Holdout Test Set ({len(X_test):,} unseen samples):")
    test_metrics = {}

    for name, pipe in fitted_pipelines.items():
        y_pred = pipe.predict(X_test)
        # Safeguard: house prices are strictly positive
        y_pred = np.clip(y_pred, a_min=10000.0, a_max=None)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        test_metrics[name] = {
            "test_mae": round(mae, 2),
            "test_rmse": round(rmse, 2),
            "test_r2": round(r2, 4),
        }

        prefix = " (*) [CHAMPION]" if name == best_model_name else "    "
        print(f"{prefix} {name:25s} | Test R2: {r2:.4f} | Test MAE: ${mae:,.2f} | Test RMSE: ${rmse:,.2f}")

    # [7] Feature Importance extraction if Random Forest
    champion_pipe = fitted_pipelines[best_model_name]
    feature_importances = []
    if hasattr(champion_pipe.named_steps["regressor"], "feature_importances_"):
        rf = champion_pipe.named_steps["regressor"]
        preprocessor = champion_pipe.named_steps["preprocessor"]
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_names = [f"zipcode_{c}" for c in cat_encoder.categories_[0]]
        all_feat_names = NUMERICAL_FEATURES + cat_names

        importances = rf.feature_importances_
        feat_imp_pairs = sorted(zip(all_feat_names, importances), key=lambda x: x[1], reverse=True)
        feature_importances = [{"feature": f, "importance": round(float(imp), 4)} for f, imp in feat_imp_pairs[:15]]

    # [8] Persist Model Artifacts & Metadata
    print(f"\n[8] Serializing Champion Model to '{REGRESSION_MODEL_PATH}'...")
    joblib.dump(champion_pipe, REGRESSION_MODEL_PATH)

    metadata = {
        "task": "Price Prediction (Regression)",
        "selected_model": best_model_name,
        "random_seed": RANDOM_SEED,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "input_features": ALL_INPUT_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_comparison": {
            name: {**cv_results[name], **test_metrics[name]} for name in candidate_models
        },
        "champion_metrics": {
            "r2": test_metrics[best_model_name]["test_r2"],
            "mae": test_metrics[best_model_name]["test_mae"],
            "rmse": test_metrics[best_model_name]["test_rmse"],
            "cv_r2": cv_results[best_model_name]["cv_r2"],
        },
        "top_feature_importances": feature_importances,
        "market": "King County (Seattle, Washington, USA)",
        "notes": "Estimates reflect historical Seattle market conditions. Not guaranteed appraisals.",
    }
    save_metadata(metadata, REGRESSION_META_PATH)
    print(f"    Metadata saved to '{REGRESSION_META_PATH}'.")

    # Quick test prediction verification
    sample_input = X_test.iloc[[0]]
    sample_actual = y_test[0]
    sample_pred = predict_price(champion_pipe, sample_input)
    print(f"\n[9] Verification Prediction Check:")
    print(f"    Actual Price   : ${sample_actual:,.2f}")
    print(f"    Estimated Price: ${sample_pred[0]:,.2f}")
    print("=" * 75)
    print(" Regression Training Completed Successfully!")
    print("=" * 75)


def predict_price(model_pipeline, input_df: pd.DataFrame) -> np.ndarray:
    """
    Safe inference function. Returns positive, finite price estimates.
    """
    preds = model_pipeline.predict(input_df)
    # Clip to sensible positive floor
    preds = np.clip(preds, a_min=10000.0, a_max=None)
    if not np.all(np.isfinite(preds)):
        raise ValueError("Model produced non-finite prediction values.")
    return preds


if __name__ == "__main__":
    train_regression_models()
