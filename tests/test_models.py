"""
test_models.py
Automated test suite for EstateIQ platform.
Validates data integrity, model loading, schema consistency, inference outputs, and edge cases.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (
    load_housing_data,
    evaluate_investment,
    get_price_category,
    DATA_PATH,
    REGRESSION_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    CLUSTERER_MODEL_PATH,
    ALL_INPUT_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    CLUSTERING_FEATURES,
    PRICE_CATEGORIES,
    BUDGET_MAX,
    MIDRANGE_MAX,
)
from train import predict_price
from classification import predict_category
from clustering import predict_cluster


@pytest.fixture(scope="module")
def sample_property_df():
    """Provides a valid single-property DataFrame matching model input schema."""
    data = {
        "bedrooms": [3],
        "bathrooms": [2.25],
        "sqft_living": [2170],
        "sqft_lot": [7242],
        "floors": [2.0],
        "waterfront": [0],
        "view": [0],
        "condition": [3],
        "grade": [7],
        "sqft_above": [2170],
        "sqft_basement": [0],
        "yr_built": [1951],
        "yr_renovated": [1991],
        "lat": [47.7210],
        "long": [-122.3190],
        "sqft_living15": [1690],
        "sqft_lot15": [7639],
        "zipcode": ["98125"],
    }
    df = pd.DataFrame(data)
    df["zipcode"] = df["zipcode"].astype(object)
    return df


@pytest.fixture(scope="module")
def sample_clustering_df():
    """Provides valid feature input for clustering model."""
    data = {
        "sqft_living": [2170],
        "bedrooms": [3],
        "bathrooms": [2.25],
        "grade": [7],
        "condition": [3],
        "price": [538000.0],
    }
    return pd.DataFrame(data)


# 1. Dataset Loading & Integrity Tests
def test_data_loading():
    """Verify that dataset loads properly and conforms to structure."""
    df = load_housing_data(DATA_PATH)
    assert not df.empty, "Dataset should not be empty"
    assert len(df) > 20000, "Dataset should contain full King County record count (>20,000)"
    assert "price" in df.columns, "Target 'price' column must exist"
    for feat in NUMERICAL_FEATURES:
        assert feat in df.columns, f"Expected numerical feature '{feat}' missing"
    assert "zipcode" in df.columns, "Categorical feature 'zipcode' missing"
    assert (df["price"] > 0).all(), "All prices should be positive numbers"


def test_missing_data_error_handling():
    """Ensure proper FileNotFoundError when dataset path is invalid."""
    with pytest.raises(FileNotFoundError):
        load_housing_data("non_existent_dataset.csv")


# 2. Regression Model Tests
def test_regression_model_prediction(sample_property_df):
    """Verify regression model loads, accepts valid inputs, and predicts positive finite price."""
    assert REGRESSION_MODEL_PATH.exists(), "Trained regression model artifact must exist"
    model = joblib.load(REGRESSION_MODEL_PATH)
    preds = predict_price(model, sample_property_df)

    assert len(preds) == 1
    price_est = preds[0]
    assert np.isfinite(price_est), "Predicted price must be finite"
    assert price_est > 50000.0, "Predicted price should be reasonably above minimum house price"
    assert price_est < 20000000.0, "Predicted price should not exceed astronomical bound"


def test_regression_missing_features_error(sample_property_df):
    """Verify regression model rejects DataFrame missing required features."""
    model = joblib.load(REGRESSION_MODEL_PATH)
    incomplete_df = sample_property_df.drop(columns=["sqft_living"])
    with pytest.raises(Exception):
        model.predict(incomplete_df)


# 3. Classification Model Tests
def test_classification_model_prediction(sample_property_df):
    """Verify classifier predicts valid price tier and probabilities."""
    assert CLASSIFIER_MODEL_PATH.exists(), "Trained classification model artifact must exist"
    classifier = joblib.load(CLASSIFIER_MODEL_PATH)

    cat, probs = predict_category(classifier, sample_property_df)
    assert cat in PRICE_CATEGORIES, f"Predicted class '{cat}' not in {PRICE_CATEGORIES}"
    assert len(probs) == len(PRICE_CATEGORIES)
    prob_sum = sum(probs.values())
    assert 0.99 <= prob_sum <= 1.01, f"Probabilities must sum to ~1.0, got {prob_sum}"


def test_price_category_thresholds():
    """Verify deterministic boundary labeling function."""
    assert get_price_category(200000) == "Budget"
    assert get_price_category(BUDGET_MAX) == "Budget"
    assert get_price_category(BUDGET_MAX + 1) == "Mid-Range"
    assert get_price_category(MIDRANGE_MAX) == "Mid-Range"
    assert get_price_category(MIDRANGE_MAX + 1) == "Premium"
    assert get_price_category(1200000) == "Premium"


# 4. Clustering Model Tests
def test_clustering_model_prediction(sample_clustering_df):
    """Verify clustering pipeline produces valid cluster IDs."""
    assert CLUSTERER_MODEL_PATH.exists(), "Trained clusterer artifact must exist"
    clusterer = joblib.load(CLUSTERER_MODEL_PATH)

    cluster_id = predict_cluster(clusterer, sample_clustering_df)
    assert isinstance(cluster_id, int), "Cluster ID must be integer"
    n_clusters = clusterer.named_steps["kmeans"].n_clusters
    assert 0 <= cluster_id < n_clusters, f"Cluster ID must be in range [0, {n_clusters - 1}]"


# 5. Feature Schema Consistency
def test_feature_schema_consistency():
    """Verify input columns list matches exactly the numerical + categorical union."""
    expected = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    actual = set(ALL_INPUT_FEATURES)
    assert expected == actual, "ALL_INPUT_FEATURES must be identical to NUMERICAL + CATEGORICAL union"
    assert "price" not in ALL_INPUT_FEATURES, "Target column must not leak into input features"
    assert "id" not in ALL_INPUT_FEATURES, "'id' column must not leak into input features"


# 6. Investment Analyzer Calculations
def test_investment_analyzer_discount():
    """Test investment analyzer detects undervalued property."""
    asking = 400000.0
    estimated = 500000.0  # Asking is 20% below estimate
    result = evaluate_investment(asking, estimated, tolerance=0.10)
    assert result["status"] == "Below model estimate"
    assert result["diff"] == -100000.0
    assert result["pct_diff"] == -20.0


def test_investment_analyzer_fair_value():
    """Test investment analyzer identifies fair market valuation within tolerance."""
    asking = 510000.0
    estimated = 500000.0  # Asking is +2% above estimate (within 10% tolerance)
    result = evaluate_investment(asking, estimated, tolerance=0.10)
    assert result["status"] == "Near model estimate"
    assert result["diff"] == 10000.0
    assert result["pct_diff"] == 2.0


def test_investment_analyzer_premium():
    """Test investment analyzer detects overvalued property."""
    asking = 650000.0
    estimated = 500000.0  # Asking is +30% above estimate
    result = evaluate_investment(asking, estimated, tolerance=0.10)
    assert result["status"] == "Above model estimate"
    assert result["diff"] == 150000.0
    assert result["pct_diff"] == 30.0

