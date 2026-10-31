"""
eda.py
Exploratory Data Analysis for EstateIQ.
Generates comprehensive summary statistics and saves analytical visualizations to eda_outputs/.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless scripts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from utils import load_housing_data, EDA_DIR, DATA_PATH


def setup_eda_directory():
    """Ensure output directory exists."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def generate_eda_report():
    """Execute complete EDA pipeline, print findings, and generate visualizations."""
    setup_eda_directory()

    print("=" * 70)
    print(" EstateIQ: Exploratory Data Analysis Pipeline")
    print("=" * 70)

    try:
        df = load_housing_data(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    # 1. Dataset Dimensions & Basic Stats
    n_rows, n_cols = df.shape
    print(f"\n[1] Dataset Dimensions: {n_rows:,} rows, {n_cols} columns")

    # 2. Missing Value & Duplicate Check
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    duplicate_rows = df.duplicated().sum()
    duplicate_ids = df["id"].duplicated().sum() if "id" in df.columns else 0

    print(f"\n[2] Data Quality Summary:")
    print(f"    - Total Missing Values: {total_missing}")
    print(f"    - Full Duplicate Rows: {duplicate_rows}")
    print(f"    - Repeated Property IDs (Re-sales): {duplicate_ids}")

    # 3. Key Descriptive Statistics
    print("\n[3] Key Feature Distributions:")
    key_cols = ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "grade", "yr_built"]
    stats_df = df[key_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    stats_df.rename(columns={"50%": "median"}, inplace=True)
    print(stats_df.to_string())

    sns.set_theme(style="whitegrid", palette="muted")

    # Chart 1: Sale Price Distribution
    print("\n[+] Generating Chart 1: Price Distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["price"] / 1000, kde=True, bins=50, color="#1f77b4", ax=ax)
    mean_val = (df["price"].mean()) / 1000
    med_val = (df["price"].median()) / 1000
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.8, label=f"Mean: ${mean_val:,.0f}k")
    ax.axvline(med_val, color="green", linestyle="-.", linewidth=1.8, label=f"Median: ${med_val:,.0f}k")
    ax.set_title("Distribution of House Sale Prices (King County, WA)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Sale Price (Thousands USD)", fontsize=11)
    ax.set_ylabel("Property Count", fontsize=11)
    ax.set_xlim(0, 3000)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "price_distribution.png", dpi=180)
    plt.close(fig)

    # Chart 2: Price vs Living Area
    print("[+] Generating Chart 2: Price vs Living Area...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x="sqft_living",
        y="price",
        hue="grade",
        palette="viridis",
        alpha=0.6,
        edgecolor=None,
        ax=ax,
    )
    sns.regplot(
        data=sample_df,
        x="sqft_living",
        y="price",
        scatter=False,
        ax=ax,
        color="red",
        line_kws={"linewidth": 2, "label": "Linear Fit"},
    )
    ax.set_title("House Price vs. Living Area (sqft) Colored by Construction Grade", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Interior Living Space (sq ft)", fontsize=11)
    ax.set_ylabel("Sale Price (USD)", fontsize=11)
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend(title="Grade (1-13)", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "price_vs_sqft_living.png", dpi=180)
    plt.close(fig)

    # Chart 3: Price by Bedroom Count
    print("[+] Generating Chart 3: Price by Bedroom Count...")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Filter for standard residential bedroom ranges (1 to 8) to avoid extreme outliers
    bed_df = df[(df["bedrooms"] >= 1) & (df["bedrooms"] <= 8)]
    sns.boxplot(data=bed_df, x="bedrooms", y="price", hue="bedrooms", palette="Blues_r", showfliers=False, ax=ax, legend=False)
    ax.set_title("Sale Price Distribution by Number of Bedrooms", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Bedrooms", fontsize=11)
    ax.set_ylabel("Sale Price (USD)", fontsize=11)
    ax.yaxis.set_major_formatter("${x:,.0f}")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "price_by_bedrooms.png", dpi=180)
    plt.close(fig)

    # Chart 4: Price by Property Grade
    print("[+] Generating Chart 4: Price by Property Grade...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="grade", y="price", hue="grade", palette="mako", showfliers=False, ax=ax, legend=False)
    ax.set_title("Sale Price vs. Construction Grade (King County System 1-13)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Construction & Design Grade Rating", fontsize=11)
    ax.set_ylabel("Sale Price (USD)", fontsize=11)
    ax.yaxis.set_major_formatter("${x:,.0f}")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "price_by_grade.png", dpi=180)
    plt.close(fig)

    # Chart 5: Correlation Heatmap
    print("[+] Generating Chart 5: Numerical Correlation Heatmap...")
    fig, ax = plt.subplots(figsize=(12, 9))
    num_cols = [
        "price",
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
        "lat",
        "long",
    ]
    corr_matrix = df[num_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Housing Features", fontsize=14, fontweight="bold", pad=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    # Chart 6: Geographic Distribution
    print("[+] Generating Chart 6: Geographic Distribution Map...")
    fig, ax = plt.subplots(figsize=(11, 8))
    geo_sample = df.sample(n=min(8000, len(df)), random_state=42)
    sc = ax.scatter(
        geo_sample["long"],
        geo_sample["lat"],
        c=geo_sample["price"] / 1000,
        cmap="plasma",
        s=12,
        alpha=0.6,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Price (Thousands USD)", fontsize=11)
    ax.set_title("Geographic Price Distribution (Latitude vs. Longitude, King County)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "geographic_distribution.png", dpi=180)
    plt.close(fig)

    print("\n" + "=" * 70)
    print(" EDA Completed Successfully! All charts saved in 'eda_outputs/':")
    print(f"  1. {EDA_DIR / 'price_distribution.png'}")
    print(f"  2. {EDA_DIR / 'price_vs_sqft_living.png'}")
    print(f"  3. {EDA_DIR / 'price_by_bedrooms.png'}")
    print(f"  4. {EDA_DIR / 'price_by_grade.png'}")
    print(f"  5. {EDA_DIR / 'correlation_heatmap.png'}")
    print(f"  6. {EDA_DIR / 'geographic_distribution.png'}")
    print("=" * 70)


if __name__ == "__main__":
    generate_eda_report()
