"""
classification.py
Machine Learning Component 2: Property Price Category Classification.
Categorizes homes into Budget, Mid-Range, or Premium market tiers based on intrinsic attributes.
"""

import sys
import time
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from utils import (
    load_housing_data,
    build_preprocessor,
    save_metadata,
    get_price_category,
    BUDGET_MAX,
    MIDRANGE_MAX,
    PRICE_CATEGORIES,
    DATA_PATH,
    MODELS_DIR,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_META_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_INPUT_FEATURES,
    TARGET_COL,
)


def train_classification_model():
    """
    Train and evaluate a Random Forest Classifier to categorize properties
    into Budget, Mid-Range, and Premium price segments.
    """
    print("=" * 75)
    print(" EstateIQ: Machine Learning Component 2 — Price Classification Pipeline")
    print("=" * 75)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[1] Loading dataset from '{DATA_PATH}'...")
    df = load_housing_data(DATA_PATH)
    print(f"    Loaded {len(df):,} property records.")

    # [2] Define Price Category Target
    print("\n[2] Engineering Classification Target:")
    print(f"    - Budget    : Price <= ${BUDGET_MAX:,.0f}")
    print(f"    - Mid-Range : ${BUDGET_MAX:,.0f} < Price <= ${MIDRANGE_MAX:,.0f}")
    print(f"    - Premium   : Price > ${MIDRANGE_MAX:,.0f}")

    y = np.array([get_price_category(p) for p in df[TARGET_COL]], dtype=object)
    class_distribution = pd.Series(y).value_counts(normalize=True).to_dict()
    print("    Class Distribution:")
    for cat in PRICE_CATEGORIES:
        count = int((pd.Series(y) == cat).sum())
        pct = class_distribution.get(cat, 0.0) * 100.0
        print(f"      * {cat:10s}: {count:,} properties ({pct:.2f}%)")

    # [3] Prevent Target Leakage: Drop 'price' and derived columns from features
    X = df[ALL_INPUT_FEATURES].copy()

    # [4] Stratified Train/Test Split
    RANDOM_SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"\n[3] Stratified Train/Test Partitioning:")
    print(f"    - Training records : {len(X_train):,} (80%)")
    print(f"    - Test records     : {len(X_test):,} (20%)")

    # [5] Build Pipeline
    print("\n[4] Constructing Preprocessing & Classifier Pipeline...")
    preprocessor = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    classifier = RandomForestClassifier(
        n_estimators=80,
        max_depth=14,
        min_samples_split=4,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    # [6] Train Model
    print("    Training Random Forest Classifier on training split...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_duration = time.time() - t0
    print(f"    Training completed in {fit_duration:.2f} seconds.")

    # [7] Evaluate Model on Unseen Holdout Test Split
    print("\n[5] Evaluating Classifier on Holdout Test Set:")
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=PRICE_CATEGORIES).tolist()
    report_dict = classification_report(y_test, y_pred, labels=PRICE_CATEGORIES, output_dict=True, zero_division=0)

    print(f"    - Test Accuracy          : {acc * 100:.2f}%")
    print(f"    - Precision (Macro)      : {prec_macro:.4f} | Weighted: {prec_weighted:.4f}")
    print(f"    - Recall (Macro)         : {rec_macro:.4f} | Weighted: {rec_weighted:.4f}")
    print(f"    - F1-Score (Macro)       : {f1_m:.4f} | Weighted: {f1_w:.4f}")

    print("\n    Per-Class Detailed Report:")
    for cat in PRICE_CATEGORIES:
        metrics = report_dict[cat]
        print(
            f"      * {cat:10s} -> Precision: {metrics['precision']:.3f} | "
            f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f} (support: {metrics['support']})"
        )

    print("\n    Confusion Matrix (Rows: Actual, Cols: Predicted):")
    print(f"                  {'  '.join([f'{c:>10s}' for c in PRICE_CATEGORIES])}")
    for idx, row in enumerate(cm):
        row_str = "  ".join([f"{val:>10d}" for val in row])
        print(f"      {PRICE_CATEGORIES[idx]:10s} [ {row_str} ]")

    # [8] Serialize Pipeline & Metadata
    print(f"\n[6] Serializing Classifier Pipeline to '{CLASSIFIER_MODEL_PATH}'...")
    joblib.dump(pipeline, CLASSIFIER_MODEL_PATH)

    metadata = {
        "task": "Price Tier Classification",
        "model_type": "Random Forest Classifier (Balanced)",
        "random_seed": RANDOM_SEED,
        "classes": PRICE_CATEGORIES,
        "thresholds": {
            "Budget": f"<= ${BUDGET_MAX:,.0f}",
            "Mid-Range": f">${BUDGET_MAX:,.0f} - ${MIDRANGE_MAX:,.0f}",
            "Premium": f"> ${MIDRANGE_MAX:,.0f}",
            "budget_max": BUDGET_MAX,
            "midrange_max": MIDRANGE_MAX,
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_macro": round(rec_macro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_macro": round(f1_m, 4),
            "f1_weighted": round(f1_w, 4),
        },
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "class_distribution": class_distribution,
        "input_features": ALL_INPUT_FEATURES,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "fit_time_seconds": round(fit_duration, 2),
    }
    save_metadata(metadata, CLASSIFIER_META_PATH)
    print(f"    Classification metadata saved to '{CLASSIFIER_META_PATH}'.")

    # Sample inference check
    sample_feat = X_test.iloc[[0]]
    sample_true = y_test[0]
    sample_pred_cat = pipeline.predict(sample_feat)[0]
    sample_proba = pipeline.predict_proba(sample_feat)[0]
    proba_dict = {cls: round(float(p), 3) for cls, p in zip(pipeline.classes_, sample_proba)}
    print(f"\n[7] Verification Prediction Check:")
    print(f"    Actual Category   : {sample_true}")
    print(f"    Predicted Category: {sample_pred_cat} (Probabilities: {proba_dict})")
    print("=" * 75)
    print(" Classification Training Completed Successfully!")
    print("=" * 75)


def predict_category(classifier_pipeline, input_df: pd.DataFrame) -> tuple[str, dict]:
    """
    Predict property price tier with class probabilities.
    """
    cat = classifier_pipeline.predict(input_df)[0]
    probs = classifier_pipeline.predict_proba(input_df)[0]
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classifier_pipeline.classes_, probs)}
    return cat, prob_dict


if __name__ == "__main__":
    train_classification_model()
