"""
utils.py
Common utilities, schemas, and helper functions for EstateIQ platform.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Project directories & paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "kc_house_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
EDA_DIR = PROJECT_ROOT / "eda_outputs"

REGRESSION_MODEL_PATH = MODELS_DIR / "house_price_model.pkl"
REGRESSION_META_PATH = MODELS_DIR / "regression_metadata.json"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "house_classifier.pkl"
CLASSIFIER_META_PATH = MODELS_DIR / "classification_metadata.json"
CLUSTERER_MODEL_PATH = MODELS_DIR / "house_clusterer.pkl"
CLUSTERER_META_PATH = MODELS_DIR / "clustering_metadata.json"

# Column definitions
TARGET_COL = "price"
DROP_COLS = ["id", "date"]

NUMERICAL_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

CATEGORICAL_FEATURES = ["zipcode"]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

CLUSTERING_FEATURES = [
    "sqft_living",
    "bedrooms",
    "bathrooms",
    "grade",
    "condition",
    "price",
]

# Classification Thresholds (USD)
# Based on King County housing market distribution
BUDGET_MAX = 380000.0
MIDRANGE_MAX = 650000.0
PRICE_CATEGORIES = ["Budget", "Mid-Range", "Premium"]


def get_price_category(price: float) -> str:
    """Classify a numeric house price into Budget, Mid-Range, or Premium."""
    if price <= BUDGET_MAX:
        return "Budget"
    elif price <= MIDRANGE_MAX:
        return "Mid-Range"
    else:
        return "Premium"


def load_housing_data(filepath: Path | str = DATA_PATH) -> pd.DataFrame:
    """
    Load and validate King County housing dataset.
    Performs integrity checks and type corrections.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{p}'. Please download 'kc_house_data.csv' "
            f"and place it into the '{p.parent}' directory."
        )

    df = pd.read_csv(p)

    required = ["price", "bedrooms", "bathrooms", "sqft_living", "grade", "yr_built", "zipcode"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing essential columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Convert zipcode to object string for clean categorical encoding across pandas versions
    df["zipcode"] = df["zipcode"].astype(str).str.strip().str.replace(".0", "", regex=False).astype(object)

    # Basic cleaning: ensure non-negative numeric entries for key features
    numeric_to_check = ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot"]
    for col in numeric_to_check:
        if col in df.columns:
            df = df[df[col] >= 0]

    return df


def build_preprocessor(
    numerical_features: list[str] = NUMERICAL_FEATURES,
    categorical_features: list[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer with median imputation and scaling
    for numeric features, and most-frequent imputation and One-Hot Encoding for categorical features.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def format_currency(val: float | int) -> str:
    """Format numeric values as USD currency strings."""
    if val is None or np.isnan(val):
        return "N/A"
    return f"${val:,.0f}"


def evaluate_investment(
    asking_price: float, estimated_price: float, tolerance: float = 0.10
) -> dict:
    """
    Compare asking price against model estimate.
    Returns status, dollar difference, and percentage variance.
    """
    if estimated_price <= 0:
        return {
            "status": "Invalid Estimate",
            "diff": 0.0,
            "pct_diff": 0.0,
            "badge_color": "secondary",
        }

    diff = asking_price - estimated_price
    pct_diff = (diff / estimated_price) * 100.0

    lower_bound = -tolerance * 100.0
    upper_bound = tolerance * 100.0

    if pct_diff < lower_bound:
        status = "Below model estimate"
        verdict = "Discount / Potential Value"
        color = "green"
    elif pct_diff > upper_bound:
        status = "Above model estimate"
        verdict = "Premium / Potential Overpricing"
        color = "red"
    else:
        status = "Near model estimate"
        verdict = "Fair Market Range"
        color = "blue"

    return {
        "asking_price": asking_price,
        "estimated_price": estimated_price,
        "diff": diff,
        "pct_diff": pct_diff,
        "status": status,
        "verdict": verdict,
        "color": color,
        "tolerance_pct": tolerance * 100,
    }


def get_feature_ranges(df: pd.DataFrame) -> dict:
    """
    Compute sensible min, max, median, and default values for UI widgets.
    """
    ranges = {}
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            s = df[col].dropna()
            ranges[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median()),
                "mean": float(s.mean()),
                "default": float(s.median()),
            }

    if "zipcode" in df.columns:
        ranges["zipcodes"] = sorted(df["zipcode"].dropna().unique().tolist())
    else:
        ranges["zipcodes"] = []

    return ranges


def save_metadata(meta: dict, filepath: Path | str) -> None:
    """Save dictionary metadata to JSON."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_metadata(filepath: Path | str) -> dict:
    """Load metadata dictionary from JSON."""
    p = Path(filepath)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
