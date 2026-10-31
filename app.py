"""
app.py
EstateIQ — Real Estate Intelligence Platform
Interactive Streamlit Dashboard for Market Analysis, Price Estimation,
Classification, and Investment Analytics.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure root path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import (
    load_housing_data,
    format_currency,
    evaluate_investment,
    get_price_category,
    get_feature_ranges,
    load_metadata,
    DATA_PATH,
    REGRESSION_MODEL_PATH,
    REGRESSION_META_PATH,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_META_PATH,
    CLUSTERER_MODEL_PATH,
    CLUSTERER_META_PATH,
    ALL_INPUT_FEATURES,
    CLUSTERING_FEATURES,
    PRICE_CATEGORIES,
    BUDGET_MAX,
    MIDRANGE_MAX,
)
from train import predict_price
from classification import predict_category
from clustering import predict_cluster


# Page configuration
st.set_page_config(
    page_title="EstateIQ — Real Estate Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Cached resource loaders
@st.cache_data(show_spinner="Loading King County Housing Dataset...")
def get_dataset():
    """Load and cache housing dataset."""
    if not DATA_PATH.exists():
        return None
    try:
        return load_housing_data(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


@st.cache_resource(show_spinner="Loading Regression Model...")
def get_regression_model():
    """Load serialized regression pipeline."""
    if not REGRESSION_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(REGRESSION_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading price prediction model: {e}")
        return None


@st.cache_resource(show_spinner="Loading Classification Model...")
def get_classifier_model():
    """Load serialized classifier pipeline."""
    if not CLASSIFIER_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(CLASSIFIER_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None


@st.cache_resource(show_spinner="Loading Clustering Model...")
def get_clusterer_model():
    """Load serialized clustering pipeline."""
    if not CLUSTERER_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(CLUSTERER_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading clustering model: {e}")
        return None


# Sidebar Navigation
st.sidebar.title("🏠 EstateIQ")
st.sidebar.caption("Real Estate Intelligence Platform")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Market Overview",
        "🔮 Property Price Predictor",
        "💼 Investment Analyzer",
        "🧠 ML Insights",
        "ℹ️ About the Project",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Market Focus**: King County, WA (Seattle Area)\n\n"
    "Estimates reflect historic educational dataset distributions and are not applicable to other markets."
)


# Header Component
def render_header():
    st.title("EstateIQ — Real Estate Intelligence")
    st.markdown(
        "*Explore housing markets. Estimate property values. Discover investment opportunities.*"
    )
    st.markdown("---")


# Check dependencies and models
df_main = get_dataset()
reg_model = get_regression_model()
clf_model = get_classifier_model()
cls_model = get_clusterer_model()

reg_meta = load_metadata(REGRESSION_META_PATH)
clf_meta = load_metadata(CLASSIFIER_META_PATH)
cls_meta = load_metadata(CLUSTERER_META_PATH)


# ==============================================================================
# SECTION A: MARKET OVERVIEW
# ==============================================================================
if nav_choice == "📊 Market Overview":
    render_header()
    st.subheader("📊 King County Housing Market Overview")

    if df_main is None:
        st.error(
            "⚠️ **Dataset file `data/kc_house_data.csv` was not found.**\n\n"
            "Please ensure the King County House Sales dataset is downloaded and saved to `data/kc_house_data.csv`."
        )
        st.stop()

    # Filters in an expander
    with st.expander("🔍 Interactive Market Filters", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            price_min = int(df_main["price"].min())
            price_max = int(df_main["price"].max())
            selected_price_range = st.slider(
                "Filter Price Range (USD)",
                min_value=price_min,
                max_value=min(price_max, 2500000),
                value=(price_min, 1500000),
                step=25000,
                format="$%d",
            )
        with col_f2:
            bed_options = sorted([int(b) for b in df_main["bedrooms"].unique() if 1 <= b <= 7])
            selected_beds = st.multiselect(
                "Select Bedrooms",
                options=bed_options,
                default=[2, 3, 4],
            )
        with col_f3:
            grade_options = sorted([int(g) for g in df_main["grade"].unique()])
            selected_grades = st.multiselect(
                "Select Construction Grade (1-13)",
                options=grade_options,
                default=[6, 7, 8, 9, 10],
            )

    # Filter DataFrame
    filtered_df = df_main[
        (df_main["price"] >= selected_price_range[0])
        & (df_main["price"] <= selected_price_range[1])
    ]
    if selected_beds:
        filtered_df = filtered_df[filtered_df["bedrooms"].isin(selected_beds)]
    if selected_grades:
        filtered_df = filtered_df[filtered_df["grade"].isin(selected_grades)]

    if filtered_df.empty:
        st.warning("No properties match the chosen filter criteria. Showing all properties.")
        filtered_df = df_main

    # KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Properties", f"{len(filtered_df):,}")
    kpi2.metric("Average Price", format_currency(filtered_df["price"].mean()))
    kpi3.metric("Median Price", format_currency(filtered_df["price"].median()))
    kpi4.metric("Avg Living Area", f"{filtered_df['sqft_living'].mean():,.0f} sqft")
    kpi5.metric(
        "Typical Layout",
        f"{filtered_df['bedrooms'].median():.0f} bed / {filtered_df['bathrooms'].median():.1f} bath",
    )

    st.markdown("### Market Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Price Distribution", "📐 Price vs. Living Space", "🏗️ Construction Grade", "🗺️ Geographic Map"]
    )

    with tab1:
        fig_dist = px.histogram(
            filtered_df,
            x="price",
            nbins=50,
            marginal="box",
            title="House Sale Price Distribution (with Boxplot)",
            labels={"price": "Sale Price (USD)"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig_dist.update_layout(xaxis_tickprefix="$", xaxis_tickformat=",.0f")
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab2:
        sample_viz = filtered_df.sample(n=min(3000, len(filtered_df)), random_state=42)
        fig_scatter = px.scatter(
            sample_viz,
            x="sqft_living",
            y="price",
            color="grade",
            color_continuous_scale="Viridis",
            trendline="ols",
            title="Sale Price vs. Living Area (sqft) with OLS Trendline",
            labels={"sqft_living": "Interior Living Area (sqft)", "price": "Sale Price (USD)", "grade": "Grade"},
            hover_data=["bedrooms", "bathrooms", "yr_built", "zipcode"],
        )
        fig_scatter.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        grade_summary = (
            filtered_df.groupby("grade")["price"].agg(["median", "mean", "count"]).reset_index()
        )
        fig_grade = px.bar(
            grade_summary,
            x="grade",
            y="median",
            text_auto=".2s",
            title="Median Property Price by King County Construction Grade (1-13)",
            labels={"grade": "King County Grade Rating", "median": "Median Price (USD)"},
            color="median",
            color_continuous_scale="Blues",
        )
        fig_grade.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        st.plotly_chart(fig_grade, use_container_width=True)

    with tab4:
        st.markdown("**Spatial Density and Valuation Across King County**")
        geo_sample = filtered_df.sample(n=min(3500, len(filtered_df)), random_state=42)
        fig_map = px.scatter(
            geo_sample,
            x="long",
            y="lat",
            color="price",
            color_continuous_scale="Plasma",
            size="sqft_living",
            size_max=12,
            opacity=0.65,
            title="Geographic Property Distribution (Latitude vs. Longitude)",
            labels={"lat": "Latitude", "long": "Longitude", "price": "Price (USD)"},
            hover_data=["bedrooms", "bathrooms", "zipcode", "price"],
        )
        st.plotly_chart(fig_map, use_container_width=True)


# ==============================================================================
# SECTION B: PROPERTY PRICE PREDICTOR
# ==============================================================================
elif nav_choice == "🔮 Property Price Predictor":
    render_header()
    st.subheader("🔮 Property Valuation Predictor")
    st.markdown(
        "Enter property characteristics to estimate fair market value using the trained Random Forest model."
    )

    if reg_model is None or clf_model is None:
        st.warning(
            "⚠️ **Trained model files not found.**\n\n"
            "Please train the models by running:\n"
            "```powershell\n"
            "python train.py\n"
            "python classification.py\n"
            "python clustering.py\n"
            "```"
        )
        st.stop()

    ranges = get_feature_ranges(df_main) if df_main is not None else {}
    zipcodes = ranges.get("zipcodes", ["98103", "98052", "98004", "98115", "98125"])

    # Form layout
    with st.form("prediction_form"):
        st.markdown("#### 1. Core Property Specifications")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=12, value=3, step=1)
            bathrooms = st.number_input(
                "Bathrooms", min_value=0.5, max_value=8.0, value=2.25, step=0.25
            )
        with col2:
            sqft_living = st.number_input(
                "Living Area (sqft)", min_value=300, max_value=14000, value=2100, step=50
            )
            sqft_lot = st.number_input(
                "Lot Area (sqft)", min_value=500, max_value=500000, value=7500, step=250
            )
        with col3:
            floors = st.selectbox("Floors", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=2)
            waterfront = st.selectbox(
                "Waterfront View",
                options=[0, 1],
                format_func=lambda x: "Yes (Waterfront)" if x == 1 else "No",
                index=0,
            )
        with col4:
            view = st.slider("Scenic View Rating", min_value=0, max_value=4, value=0)
            condition = st.slider("Condition (1=Poor, 5=Excellent)", min_value=1, max_value=5, value=3)

        st.markdown("#### 2. Construction Quality & Details")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            grade = st.slider(
                "Building Grade (1-13)",
                min_value=1,
                max_value=13,
                value=7,
                help="King County construction grade: 1-3 substandard, 7 average, 11-13 custom luxury.",
            )
        with col6:
            sqft_above = st.number_input(
                "Above Ground Area (sqft)",
                min_value=300,
                max_value=10000,
                value=min(sqft_living, 1800),
                step=50,
            )
            sqft_basement = max(0, sqft_living - sqft_above)
            st.caption(f"Calculated Basement: **{sqft_basement} sqft**")
        with col7:
            yr_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=1980, step=1)
            yr_renovated = st.number_input(
                "Year Renovated (0 if never)", min_value=0, max_value=2024, value=0, step=1
            )
        with col8:
            selected_zip = st.selectbox("Zip Code", options=zipcodes, index=min(10, len(zipcodes) - 1))

        # Spatial defaults from zipcode median if available
        if df_main is not None:
            zip_df = df_main[df_main["zipcode"] == str(selected_zip)]
            default_lat = float(zip_df["lat"].median()) if not zip_df.empty else 47.56
            default_long = float(zip_df["long"].median()) if not zip_df.empty else -122.21
            default_living15 = int(zip_df["sqft_living15"].median()) if not zip_df.empty else sqft_living
            default_lot15 = int(zip_df["sqft_lot15"].median()) if not zip_df.empty else sqft_lot
        else:
            default_lat, default_long, default_living15, default_lot15 = 47.56, -122.21, sqft_living, sqft_lot

        submitted = st.form_submit_button("⚡ Estimate Property Price", use_container_width=True)

    if submitted:
        input_data = {
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "sqft_living": [sqft_living],
            "sqft_lot": [sqft_lot],
            "floors": [floors],
            "waterfront": [waterfront],
            "view": [view],
            "condition": [condition],
            "grade": [grade],
            "sqft_above": [sqft_above],
            "sqft_basement": [sqft_basement],
            "yr_built": [yr_built],
            "yr_renovated": [yr_renovated],
            "lat": [default_lat],
            "long": [default_long],
            "sqft_living15": [default_living15],
            "sqft_lot15": [default_lot15],
            "zipcode": [str(selected_zip)],
        }
        input_df = pd.DataFrame(input_data)
        input_df["zipcode"] = input_df["zipcode"].astype(object)

        try:
            pred_price = predict_price(reg_model, input_df)[0]
            pred_cat, cat_probs = predict_category(clf_model, input_df)

            st.success("### Property Valuation Result")
            res_col1, res_col2 = st.columns([1.5, 1])

            with res_col1:
                st.metric("Estimated Market Value", format_currency(pred_price))
                st.caption(
                    f"Predicted Value: **{format_currency(pred_price)}** (Based on trained Random Forest Regressor)"
                )

            with res_col2:
                cat_colors = {"Budget": "🟢 Budget", "Mid-Range": "🔵 Mid-Range", "Premium": "🟣 Premium"}
                st.metric("Price Category Tier", cat_colors.get(pred_cat, pred_cat))
                st.caption(
                    f"Tier Confidence: {cat_probs.get(pred_cat, 0.0) * 100:.1f}% ({pred_cat})"
                )

            # Property insights breakdown
            st.markdown("#### 📋 Key Feature Impact Analysis")
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                st.markdown(
                    f"- **Living Area**: {sqft_living:,} sqft (${pred_price / max(1, sqft_living):.0f}/sqft)\n"
                    f"- **Grade Rating**: {grade}/13 ({'High/Luxury' if grade >= 9 else 'Standard/Average' if grade >= 7 else 'Budget'})\n"
                    f"- **Zip Code**: `{selected_zip}`"
                )
            with exp_col2:
                st.markdown(
                    f"- **Waterfront**: {'Yes (+Significant Value Premium)' if waterfront == 1 else 'None'}\n"
                    f"- **View Rating**: {view}/4\n"
                    f"- **Age**: {2024 - yr_built} years (Built {yr_built})"
                )
            with exp_col3:
                st.markdown(
                    f"- **Condition**: {condition}/5\n"
                    f"- **Renovated**: {'Yes (' + str(yr_renovated) + ')' if yr_renovated > 0 else 'No'}\n"
                    f"- **Basement**: {sqft_basement:,} sqft"
                )

            st.info(
                "ℹ️ **Legal & Practical Appraisal Disclaimer**: This valuation is generated by a statistical machine learning "
                "model trained on historic sales in King County, WA. It is an algorithmic estimation and is not a certified real estate appraisal. "
                "Local property tax assessments, micro-neighborhood condition, and property title status must be evaluated independently."
            )
        except Exception as e:
            st.error(f"Inference Error: {e}")


# ==============================================================================
# SECTION C: INVESTMENT ANALYZER
# ==============================================================================
elif nav_choice == "💼 Investment Analyzer":
    render_header()
    st.subheader("💼 Real Estate Investment & Asking Price Analyzer")
    st.markdown(
        "Compare an owner's asking price against the machine learning estimated fair market value to identify potential discounts or overpricing."
    )

    if reg_model is None:
        st.warning("⚠️ Prediction model not found. Please train models first using `python train.py`.")
        st.stop()

    ranges = get_feature_ranges(df_main) if df_main is not None else {}
    zipcodes = ranges.get("zipcodes", ["98103", "98052", "98004", "98125"])

    col_inv_left, col_inv_right = st.columns([1, 1.2])

    with col_inv_left:
        st.markdown("#### Asking Price & Deal Parameters")
        asking_price = st.number_input(
            "Property Asking Price (USD)",
            min_value=50000,
            max_value=10000000,
            value=520000,
            step=10000,
            format="%d",
        )
        tolerance_pct = st.slider(
            "Fair Value Tolerance Margin (± %)",
            min_value=3,
            max_value=20,
            value=10,
            step=1,
            help="Properties within this margin are treated as Near Model Estimate (Fair Market Range).",
        )

        st.markdown("#### Property Attributes")
        i_beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        i_baths = st.number_input("Bathrooms", min_value=1.0, max_value=8.0, value=2.0, step=0.25)
        i_sqft = st.number_input("Living Area (sqft)", min_value=400, max_value=12000, value=1950, step=50)
        i_grade = st.slider("Construction Grade (1-13)", min_value=1, max_value=13, value=7)
        i_zip = st.selectbox("Zip Code", options=zipcodes, index=0)
        analyze_btn = st.button("📊 Evaluate Asking Price", use_container_width=True)

    with col_inv_right:
        st.markdown("#### Investment Analysis Output")
        if analyze_btn:
            # Prepare feature input
            zip_df = df_main[df_main["zipcode"] == str(i_zip)] if df_main is not None else pd.DataFrame()
            d_lat = float(zip_df["lat"].median()) if not zip_df.empty else 47.56
            d_long = float(zip_df["long"].median()) if not zip_df.empty else -122.21
            d_sqft_above = int(i_sqft * 0.85)
            d_sqft_base = i_sqft - d_sqft_above

            inv_input = pd.DataFrame(
                {
                    "bedrooms": [i_beds],
                    "bathrooms": [i_baths],
                    "sqft_living": [i_sqft],
                    "sqft_lot": [7500],
                    "floors": [1.5],
                    "waterfront": [0],
                    "view": [0],
                    "condition": [3],
                    "grade": [i_grade],
                    "sqft_above": [d_sqft_above],
                    "sqft_basement": [d_sqft_base],
                    "yr_built": [1975],
                    "yr_renovated": [0],
                    "lat": [d_lat],
                    "long": [d_long],
                    "sqft_living15": [i_sqft],
                    "sqft_lot15": [7500],
                    "zipcode": [str(i_zip)],
                }
            )
            inv_input["zipcode"] = inv_input["zipcode"].astype(object)

            est_price = predict_price(reg_model, inv_input)[0]
            eval_res = evaluate_investment(
                asking_price, est_price, tolerance=tolerance_pct / 100.0
            )

            # Metric cards
            m1, m2 = st.columns(2)
            m1.metric("Asking Price", format_currency(asking_price))
            m2.metric(
                "Model Estimated Value",
                format_currency(est_price),
                delta=f"{eval_res['pct_diff']:+.1f}%",
                delta_color="inverse",
            )

            diff_val = eval_res["diff"]
            pct_val = eval_res["pct_diff"]

            if eval_res["status"] == "Below model estimate":
                st.success(
                    f"### 🏷️ Status: {eval_res['status']}\n"
                    f"**Verdict**: {eval_res['verdict']}\n\n"
                    f"The asking price is **{format_currency(abs(diff_val))} ({abs(pct_val):.1f}%) below** "
                    f"the statistical model estimate. This property may represent an attractive purchasing opportunity."
                )
            elif eval_res["status"] == "Above model estimate":
                st.error(
                    f"### ⚠️ Status: {eval_res['status']}\n"
                    f"**Verdict**: {eval_res['verdict']}\n\n"
                    f"The asking price is **{format_currency(diff_val)} ({pct_val:+.1f}%) above** "
                    f"the model estimate. The listing appears priced at a premium relative to comparable homes."
                )
            else:
                st.info(
                    f"### ⚖️ Status: {eval_res['status']}\n"
                    f"**Verdict**: {eval_res['verdict']}\n\n"
                    f"The asking price is within the configurable **±{tolerance_pct}%** tolerance band "
                    f"of the estimated fair value (${est_price:,.0f})."
                )

            # Visual gauge comparison
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=asking_price,
                    delta={"reference": est_price, "valueformat": "$,.0f"},
                    title={"text": "Asking Price vs. Model Estimate (USD)"},
                    gauge={
                        "axis": {"range": [est_price * 0.6, est_price * 1.4], "tickprefix": "$"},
                        "bar": {"color": "#1f77b4"},
                        "steps": [
                            {"range": [est_price * 0.6, est_price * (1 - tolerance_pct / 100)], "color": "#d4edda"},
                            {
                                "range": [
                                    est_price * (1 - tolerance_pct / 100),
                                    est_price * (1 + tolerance_pct / 100),
                                ],
                                "color": "#cce5ff",
                            },
                            {"range": [est_price * (1 + tolerance_pct / 100), est_price * 1.4], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": est_price,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            st.info("Adjust the asking price and property parameters on the left, then click **Evaluate Asking Price**.")

        st.warning(
            "⚠️ **Responsible Investment Consideration**: Comparison against model estimates does not guarantee financial profit "
            "or absence of investment risk. Real estate decisions must account for property tax liens, HOA dues, structural inspection reports, "
            "financing interest rates, insurance, tenant occupancy, and broader regional economic cycles."
        )


# ==============================================================================
# SECTION D: ML INSIGHTS
# ==============================================================================
elif nav_choice == "🧠 ML Insights":
    render_header()
    st.subheader("🧠 Machine Learning Performance & Model Insights")
    st.markdown(
        "Transparent evaluation metrics, model comparisons, and Viva explanations for all three machine learning tasks."
    )

    t_reg, t_clf, t_cls, t_viva = st.tabs(
        ["1️⃣ Regression Analysis", "2️⃣ Classification Analysis", "3️⃣ Clustering Analysis", "🎓 Viva Concept Guide"]
    )

    with t_reg:
        st.markdown("#### Regression Model Comparison")
        if reg_meta and "model_comparison" in reg_meta:
            comp_data = []
            for m_name, m_dict in reg_meta["model_comparison"].items():
                comp_data.append(
                    {
                        "Model Algorithm": m_name,
                        "CV R² (5-Fold)": f"{m_dict.get('cv_r2', 0):.4f}",
                        "CV MAE": f"${m_dict.get('cv_mae', 0):,.2f}",
                        "Test R² (Holdout)": f"{m_dict.get('test_r2', 0):.4f}",
                        "Test MAE (USD)": f"${m_dict.get('test_mae', 0):,.2f}",
                        "Test RMSE (USD)": f"${m_dict.get('test_rmse', 0):,.2f}",
                    }
                )
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True)

            champ_name = reg_meta.get("selected_model", "Random Forest Regressor")
            st.success(
                f"🏆 **Champion Model**: `{champ_name}` was selected based on cross-validation R² performance. "
                f"It achieved an **R² of {reg_meta['champion_metrics']['r2']:.4f}** and an **MAE of ${reg_meta['champion_metrics']['mae']:,.2f}** "
                f"on the unseen holdout test partition."
            )

            if "top_feature_importances" in reg_meta and reg_meta["top_feature_importances"]:
                st.markdown("##### Top Predictive Feature Importances (Random Forest)")
                feat_df = pd.DataFrame(reg_meta["top_feature_importances"])
                fig_imp = px.bar(
                    feat_df.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 15 Predictive Features by Gini Importance",
                    labels={"importance": "Relative Importance", "feature": "Feature"},
                    color="importance",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Run `python train.py` to populate regression metadata.")

    with t_clf:
        st.markdown("#### Price Tier Classification Performance")
        if clf_meta and "metrics" in clf_meta:
            m = clf_meta["metrics"]
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Overall Accuracy", f"{m['accuracy'] * 100:.2f}%")
            cm2.metric("Precision (Macro)", f"{m['precision_macro']:.4f}")
            cm3.metric("Recall (Macro)", f"{m['recall_macro']:.4f}")
            cm4.metric("F1-Score (Macro)", f"{m['f1_macro']:.4f}")

            st.markdown("##### Confusion Matrix (Holdout Test Split)")
            cm_matrix = clf_meta["confusion_matrix"]
            labels = clf_meta["classes"]

            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Category", y="Actual Category", color="Count"),
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix — Price Category Classification",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("##### Documented Classification Thresholds")
            thresh_data = [
                {"Category Tier": "Budget", "Price Threshold": f"<= ${BUDGET_MAX:,.0f}", "Description": "Lower tertile of housing sales"},
                {"Category Tier": "Mid-Range", "Price Threshold": f"${BUDGET_MAX:,.0f} - ${MIDRANGE_MAX:,.0f}", "Description": "Middle suburban market properties"},
                {"Category Tier": "Premium", "Price Threshold": f"> ${MIDRANGE_MAX:,.0f}", "Description": "Upper quartile & luxury waterfront/estates"},
            ]
            st.table(pd.DataFrame(thresh_data))
        else:
            st.info("Run `python classification.py` to populate classification metadata.")

    with t_cls:
        st.markdown("#### K-Means Property Clustering & Market Segmentation")
        if cls_meta and "cluster_profiles" in cls_meta:
            st.markdown(f"**Optimal Clusters Selected**: **K = {cls_meta['selected_k']}**")
            st.caption(
                f"Selected based on Silhouette Score separation ({cls_meta['silhouette_score']:.4f}) and Elbow Inertia analysis."
            )

            # Candidate K table
            if "k_evaluations" in cls_meta:
                k_data = []
                for k_str, k_info in cls_meta["k_evaluations"].items():
                    k_data.append(
                        {
                            "Candidate K": int(k_str),
                            "Silhouette Score": f"{k_info['silhouette_score']:.4f}",
                            "Inertia (SSE)": f"{k_info['inertia']:,.1f}",
                            "Status": "Selected" if int(k_str) == cls_meta["selected_k"] else "Evaluated",
                        }
                    )
                st.table(pd.DataFrame(k_data))

            st.markdown("##### Market Persona Profiles")
            profiles_df = pd.DataFrame(cls_meta["cluster_profiles"])
            display_profiles = profiles_df[
                [
                    "name",
                    "count",
                    "percentage",
                    "median_price",
                    "mean_sqft_living",
                    "mean_bedrooms",
                    "mean_bathrooms",
                    "mean_grade",
                ]
            ].copy()
            display_profiles.rename(
                columns={
                    "name": "Market Persona",
                    "count": "Property Count",
                    "percentage": "Share (%)",
                    "median_price": "Median Price ($)",
                    "mean_sqft_living": "Mean Sqft",
                    "mean_bedrooms": "Beds",
                    "mean_bathrooms": "Baths",
                    "mean_grade": "Grade",
                },
                inplace=True,
            )
            display_profiles["Median Price ($)"] = display_profiles["Median Price ($)"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display_profiles, use_container_width=True)

            # Interactive Cluster Scatter Plot
            if df_main is not None and cls_model is not None:
                st.markdown("##### Interactive Cluster Distribution Visualization")
                sample_cls = df_main[CLUSTERING_FEATURES].sample(n=min(2500, len(df_main)), random_state=42).copy()
                sample_cls["cluster"] = cls_model.predict(sample_cls)
                persona_map = {int(k): v for k, v in cls_meta.get("persona_map", {}).items()}
                sample_cls["persona"] = sample_cls["cluster"].map(persona_map)

                fig_cls = px.scatter(
                    sample_cls,
                    x="sqft_living",
                    y="price",
                    color="persona",
                    title="Property Clusters: Living Area vs. Price by Persona",
                    labels={"sqft_living": "Living Space (sqft)", "price": "Price (USD)", "persona": "Market Segment"},
                    hover_data=["bedrooms", "bathrooms", "grade"],
                )
                fig_cls.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
                st.plotly_chart(fig_cls, use_container_width=True)
        else:
            st.info("Run `python clustering.py` to populate clustering metadata.")

    with t_viva:
        st.markdown("#### 🎓 Viva / Interview Preparation Guide")
        st.markdown("Clear, plain-language answers to common technical questions asked by examiners.")

        with st.expander("Q1: What is the difference between MAE and RMSE in regression?"):
            st.markdown(
                "**Answer**: **Mean Absolute Error (MAE)** calculates the average absolute dollar magnitude of errors. "
                "Every error is penalized linearly. **Root Mean Squared Error (RMSE)** squares the errors before averaging and taking "
                "the square root. Because it squares individual residuals, RMSE severely penalizes large outlier errors. "
                "In house price prediction, RMSE is usually higher than MAE when luxury home valuation errors exist."
            )

        with st.expander("Q2: What does the R² score represent?"):
            st.markdown(
                "**Answer**: **R² (Coefficient of Determination)** measures the proportion of variance in the target variable (price) "
                "that is predictable from the independent features. An R² of 0.85 means 85% of price variability is explained by our model, "
                "while 15% is due to unobserved factors (such as interior finishes, seller motivation, or specific view angles)."
            )

        with st.expander("Q3: How was target leakage prevented in the price predictor?"):
            st.markdown(
                "**Answer**: Target leakage happens when features containing information about the target are included in the model input. "
                "We prevented leakage by:\n"
                "1. Strictly dropping `price` and any derived pricing labels from the feature matrix `X`.\n"
                "2. Excluding non-predictive administrative identifiers like `id`.\n"
                "3. Fitting imputers and scalers exclusively on the training split inside a `Pipeline` before transforming test data."
            )

        with st.expander("Q4: Why use Stratified train/test splitting in classification?"):
            st.markdown(
                "**Answer**: In classification, property classes (Budget, Mid-Range, Premium) may have differing proportions. "
                "**Stratified splitting** guarantees that both the training set and testing set maintain the exact same percentage "
                "distribution of classes as the original population, preventing train/test distribution skew."
            )

        with st.expander("Q5: What is the Silhouette Score in clustering?"):
            st.markdown(
                "**Answer**: The **Silhouette Score** ranges from -1 to +1 and assesses clustering quality by comparing intra-cluster cohesion "
                "(how close each point is to other points in its assigned cluster) against inter-cluster separation (how far it is from points in "
                "the nearest neighboring cluster). Scores above 0.20 indicate meaningful structural separation."
            )

        with st.expander("Q6: Why did you choose Random Forest over simple Linear Regression?"):
            st.markdown(
                "**Answer**: Linear Regression assumes a strictly linear relationship between features and price. However, housing prices feature "
                "non-linear interactions (e.g., adding a bathroom in a 1,000 sqft home has a different impact than in a 5,000 sqft mansion; "
                "waterfront frontage exponentially escalates value for high-grade homes). Random Forest captures these non-linearities and feature "
                "interactions without manual polynomial feature engineering."
            )


# ==============================================================================
# SECTION E: ABOUT THE PROJECT
# ==============================================================================
elif nav_choice == "ℹ️ About the Project":
    render_header()
    st.subheader("ℹ️ About EstateIQ — Real Estate Intelligence")

    st.markdown(
        """
### Project Objective
EstateIQ was conceived and developed as a comprehensive, end-to-end Machine Learning demonstration platform 
for an undergraduate BTech Information Technology curriculum. It shows practical implementation of the complete ML lifecycle:
- Exploratory Data Analysis & Statistical profiling
- Regression for continuous valuation
- Supervised classification for property tier ranking
- Unsupervised K-Means clustering for market segmentation
- Financial comparison & investment analytics

---

### Dataset Source & Geographic Coverage
- **Dataset**: King County House Sales (`kc_house_data.csv`).
- **Region**: King County, Washington, United States (including Seattle, Bellevue, Redmond, Kirkland, and surrounding municipalities).
- **Timeframe**: Historic residential transactions recorded between May 2014 and May 2015.
- **Records**: 21,613 property sales with 21 attributes (living area, bedrooms, construction grade, coordinates, etc.).

> [!IMPORTANT]
> **Geographic & Market Applicability Note**:  
> Because this dataset reflects historic market dynamics in Seattle, Washington (US dollars, square feet, King County building codes), 
> **the resulting estimates and thresholds are specific to this market and are NOT directly applicable to Indian real estate markets** 
> (e.g., Mumbai, Delhi, Bengaluru) without localized retraining on Indian land registry and transaction records.

---

### Technology Stack
- **Language**: Python 3.10+
- **Data Engineering**: pandas, NumPy
- **Machine Learning**: scikit-learn (Pipelines, Imputers, Random Forest, K-Means, Ridge)
- **Visualization**: Plotly Interactive Charts, Matplotlib, Seaborn
- **Web Dashboard**: Streamlit
- **Testing**: pytest
- **Model Persistence**: joblib, JSON metadata

---

### Ethical & Responsible AI Considerations
1. **Algorithmic Appraisal Limitation**: Machine learning models reflect patterns in training data and cannot inspect physical structural foundations, water damage, or legal title disputes.
2. **Fair Housing**: The dataset does not contain demographic or protected class data, focusing strictly on physical and geographic attributes.
3. **No Financial Guarantee**: Investment flags ("Below model estimate") indicate statistical variance, not guaranteed capital appreciation or rental return.
"""
    )

