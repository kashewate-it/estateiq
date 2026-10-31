# EstateIQ — Real Estate Intelligence Platform

> **Interactive Machine Learning & Market Intelligence Dashboard for Real Estate Analytics**  
> *Developed for BTech Information Technology — Practical Machine Learning Capstone Demonstration*

---

## 1. What EstateIQ Is

**EstateIQ** is an interactive, beginner-friendly machine learning and data analytics application designed to analyze housing markets, estimate real estate valuations, classify property price tiers, segment market demographics, and assist buyers and investors with quantitative property assessments.

Built with **Python**, **pandas**, **scikit-learn**, **Plotly**, and **Streamlit**, EstateIQ allows users to:
1. **Explore Housing Market Trends**: View descriptive statistics, spatial price heatmaps, and price-to-area distributions across King County, Washington.
2. **Estimate Property Prices**: Predict real estate sale values with a trained Random Forest Regressor using verified property attributes.
3. **Classify Price Categories**: Deterministically and probabilistically classify homes into **Budget**, **Mid-Range**, and **Premium** market tiers.
4. **Segment Property Markets**: Uncover natural market clusters (Starter, Suburban, Luxury) using unsupervised K-Means clustering.
5. **Analyze Deal Valuations**: Compare owner asking prices against reference algorithmic valuations to detect overpricing or potential investment discounts.
6. **Inspect Machine Learning Metrics**: Transparently examine Cross-Validation R², MAE, RMSE, Confusion Matrices, and Silhouette Scores.

---

## 2. Why the Project Was Created

Real estate transactions represent some of the highest-value financial decisions made by consumers and investors. However, property valuation is often non-transparent, subjective, and prone to human bias or emotional bidding.

EstateIQ was developed to show how a computer science / IT student can leverage machine learning pipelines to:
- Transform raw, messy real-world housing sales into clean feature matrices.
- Prevent **target leakage** and data snooping in predictive systems.
- Benchmark classical linear models against non-linear ensemble algorithms.
- Deliver production-grade machine learning models through a modern interactive web dashboard.

---

## 3. Features & Visual Walkthrough

- **Market Overview Dashboard**: Real-time KPI metric cards (Mean/Median price, typical layout, total inventory), dynamic sliders for price/bedroom filtering, and interactive Plotly visualizations.
- **Valuation Predictor**: Multi-column inputs for 18 property attributes (bedrooms, bathrooms, living area, condition, building grade, year built, zipcode, coordinates) delivering instantaneous price estimates in USD.
- **Investment Analyzer**: Deal evaluation engine computing delta dollars and percentage spread against asking prices with configurable tolerance bands (±10%).
- **Model Explainability & Viva Guide**: In-depth explanations of regression errors, confusion matrices, and cluster silhouettes designed for examination viva readiness.

*(Screenshots can be captured directly when launching `streamlit run app.py` on your Windows desktop).*

---

## 4. Project Structure

```text
EstateIQ/
├── app.py                      # Interactive Streamlit Web Application
├── train.py                    # Component 1: Regression training & model selection
├── classification.py           # Component 2: Price tier classification pipeline
├── clustering.py               # Component 3: K-Means market segmentation pipeline
├── eda.py                      # Exploratory Data Analysis & static chart generator
├── utils.py                    # Schemas, feature definitions, data loader & helpers
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── README.md                   # Comprehensive technical documentation & Viva guide
├── .gitignore                  # Git ignore rules for venv, cache, and temporary files
├── data/
│   └── kc_house_data.csv       # King County House Sales authentic dataset (21,613 records)
├── models/
│   ├── house_price_model.pkl   # Serialized champion Random Forest Regressor pipeline
│   ├── regression_metadata.json# Regression CV and holdout test metrics
│   ├── house_classifier.pkl    # Serialized Random Forest Classifier pipeline
│   ├── classification_metadata.json # Classification report and confusion matrix
│   ├── house_clusterer.pkl     # Serialized K-Means clustering pipeline
│   └── clustering_metadata.json# Cluster personas, silhouettes, and centroids
├── eda_outputs/                # High-resolution analytical charts
│   ├── price_distribution.png
│   ├── price_vs_sqft_living.png
│   ├── price_by_bedrooms.png
│   ├── price_by_grade.png
│   ├── correlation_heatmap.png
│   └── geographic_distribution.png
└── tests/
    ├── __init__.py
    └── test_models.py          # Pytest automated test suite (11 test cases)
```

---

## 5. Dataset Source & Column Dictionary

EstateIQ utilizes the **King County House Sales dataset** (`kc_house_data.csv`), covering residential home sales in King County, Washington (Seattle, Bellevue, Redmond, Kirkland, and surrounding cities) recorded between May 2014 and May 2015.

- **Total Rows**: 21,613 properties
- **Total Columns**: 21 columns
- **Public Educational Distribution**: University of Washington / Kaggle King County Housing Dataset.

### Column Descriptions

| Column | Data Type | Description | Role in EstateIQ |
| :--- | :--- | :--- | :--- |
| `id` | Integer | Unique identifier for property | Excluded (prevents target leakage) |
| `date` | String | Timestamp of property sale | Excluded (prevents temporal leakage) |
| `price` | Float | Final property sale price in USD | **Target Variable** (Regression & Classification) |
| `bedrooms` | Integer | Number of bedrooms (range: 1 - 33) | Input Feature |
| `bathrooms` | Float | Number of bathrooms (fractional for half/three-quarter baths) | Input Feature |
| `sqft_living` | Integer | Interior living space in square feet | Input Feature (Primary driver) |
| `sqft_lot` | Integer | Total parcel land area in square feet | Input Feature |
| `floors` | Float | Number of building levels (1.0, 1.5, 2.0, etc.) | Input Feature |
| `waterfront` | Binary | 1 if property fronts Lake Washington / Puget Sound; 0 otherwise | Input Feature |
| `view` | Integer | Rating of scenic view quality (0 = None to 4 = Panoramic) | Input Feature |
| `condition` | Integer | Overall structural maintenance condition (1 = Poor to 5 = Pristine) | Input Feature |
| `grade` | Integer | King County construction & design quality scale (1-13) | Input Feature (High correlation with price) |
| `sqft_above` | Integer | Living space above ground level (sqft) | Input Feature |
| `sqft_basement` | Integer | Living space below ground level (sqft) | Input Feature |
| `yr_built` | Integer | Year house was originally erected (1900 - 2015) | Input Feature |
| `yr_renovated` | Integer | Year of last structural renovation (0 if never renovated) | Input Feature |
| `zipcode` | Categorical | 5-digit US postal code (70 distinct King County zipcodes) | Categorical Input (One-Hot Encoded) |
| `lat` | Float | Latitude coordinate | Spatial Input Feature |
| `long` | Float | Longitude coordinate | Spatial Input Feature |
| `sqft_living15` | Integer | Average living area of nearest 15 neighboring homes | Neighborhood Context Feature |
| `sqft_lot15` | Integer | Average lot area of nearest 15 neighboring homes | Neighborhood Context Feature |

> **Geographic Boundary Warning**: Because the dataset reflects King County, Washington, valuations are in USD ($) and calibrated to Seattle suburban density, local zoning codes, and Pacific Northwest construction standards. **These models are not directly applicable to Indian real estate markets** without localized retraining on Indian municipal land registries.

---

## 6. Machine Learning Algorithms Explained

### 1. Linear Regression (Baseline)
Models the relationship between continuous features and property price as a linear equation:
$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$
- **Pros**: Fast, highly interpretable coefficients.
- **Cons**: Cannot capture complex non-linear interactions (e.g., the synergistic effect of having a high building grade *combined* with a waterfront view).

### 2. Ridge Regression ($L_2$ Regularization)
Minimizes the sum of squared residuals plus a penalty proportional to the square of coefficient magnitudes:
$$\mathcal{L}_{\text{Ridge}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2$$
- Prevents multicollinearity instability among correlated variables (such as `sqft_living` and `sqft_above`).

### 3. Random Forest Regressor (Champion Model)
An ensemble learning technique that trains dozens of unpruned Decision Trees on bootstrapped data subsets (Bagging) and averages their predictions:
- Captures high-order non-linearities and geographic boundaries naturally.
- Substantially reduces prediction variance compared to single decision trees.
- Selected as our champion model with an **$R^2$ of 0.8529** and an **MAE of $74,243**.

### 4. Random Forest Classifier
Classifies properties into **Budget**, **Mid-Range**, and **Premium** price tiers:
- Uses `class_weight="balanced"` to counteract class distribution differences.
- Achieved **83.99% overall accuracy** on the holdout test set with balanced precision and recall.

### 5. K-Means Clustering
An unsupervised clustering algorithm that partitions $N$ properties into $K$ clusters by minimizing within-cluster inertia (sum of squared Euclidean distances to the centroid):
$$\mathcal{J} = \sum_{k=1}^{K} \sum_{x \in S_k} || x - \mu_k ||^2$$
- Groups properties into three market personas: **Budget & Starter**, **Suburban Family**, and **Executive Luxury**.

---

## 7. Machine Learning Concepts & Preprocessing

### Target Leakage Prevention
Target leakage occurs when an algorithm is inadvertently trained with features that directly reveal the target or would not be available at inference time. In EstateIQ:
1. `price` is strictly removed from feature matrix $X$ before model fitting.
2. The classification label is derived from `price`, but `price` is completely dropped from classifier inputs.
3. Administrative variables (`id`) and post-sale timestamps (`date`) are discarded.
4. Preprocessing transformers (imputers and scalers) are strictly fitted on the training split only inside a `scikit-learn` `Pipeline`.

### Feature Preprocessing Pipelines
- **Numerical Features** (17 features):
  - `SimpleImputer(strategy="median")`: Robust to extreme skewness.
  - `StandardScaler()`: Standardizes variables to zero mean and unit variance.
- **Categorical Features** (`zipcode`):
  - `SimpleImputer(strategy="most_frequent")`.
  - `OneHotEncoder(handle_unknown="ignore")`: Transforms 70 postal codes into binary indicator columns without imposing an artificial ordinal relationship.

### Train / Test Splitting Methodology
- **Regression**: 80% training (17,290 records), 20% holdout testing (4,323 records) with fixed seed (`random_state=42`).
- **Model Selection**: 3-Fold Cross-Validation on the training partition to select the champion model without test-set snooping bias.
- **Classification**: Stratified 80/20 train/test split ensuring identical class proportions across both splits.

---

## 8. Evaluation Metrics Obtained

### A. Regression Model Comparison

| Model | 3-Fold CV $R^2$ | 3-Fold CV MAE | Holdout Test $R^2$ | Holdout Test MAE | Holdout Test RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.8033 | \$96,086.51 | 0.8074 | \$98,581.32 | \$170,645.62 |
| **Ridge Regression ($\alpha=10$)** | 0.8011 | \$96,371.46 | 0.8048 | \$98,911.99 | \$171,801.83 |
| **Random Forest Regressor** *(Champion)* | **0.8706** | **\$70,812.63** | **0.8529** | **\$74,243.39** | **\$149,108.29** |

### B. Classification Performance (Price Tiers)

- **Overall Accuracy**: **83.99%**
- **Macro Precision**: **0.8425** | Weighted: **0.8417**
- **Macro Recall**: **0.8425** | Weighted: **0.8399**
- **Macro F1-Score**: **0.8422** | Weighted: **0.8405**

#### Confusion Matrix (Holdout Test Split: 4,323 Properties)

| Actual Tier \ Predicted Tier | Budget | Mid-Range | Premium | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Budget** ($\le \$380\text{k}$) | **1,374** | 227 | 0 | **85.8%** |
| **Mid-Range** ($\$380\text{k} - \$650\text{k}$) | 148 | **1,363** | 169 | **81.1%** |
| **Premium** ($> \$650\text{k}$) | 3 | 145 | **894** | **85.8%** |
| **Precision** | **90.1%** | **78.6%** | **84.1%** | — |

*Notice that zero Budget properties were erroneously classified as Premium, and only 3 Premium properties were classified as Budget.*

### C. Property Clustering Segmentation ($K=3$)

- **Evaluated K Silhouette Scores**: $K=2: 0.3229$, $K=3: 0.2402$, $K=4: 0.2478$, $K=5: 0.2363$, $K=6: 0.2365$.
- **Selected K**: **$K=3$** (Justified by domain need for distinct Starter, Mid-Market, and Luxury tiers).

| Cluster ID | Market Persona | Share (%) | Median Price | Mean Sqft | Mean Beds | Mean Baths | Mean Grade |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Budget & Starter Homes** | 41.02% | \$340,000 | 1,369 sqft | 2.8 | 1.5 | 6.8 / 13 |
| **1** | **Mid-Market Suburban Residences** | 47.02% | \$499,950 | 2,270 sqft | 3.7 | 2.4 | 7.9 / 13 |
| **2** | **Executive & Luxury Estates** | 11.97% | \$1,000,000 | 3,768 sqft | 4.2 | 3.2 | 9.7 / 13 |

---

## 9. Installation & Execution Guide (Windows PowerShell)

Follow these exact steps in **Windows PowerShell**:

### Step 1: Open Project Directory
```powershell
cd c:\Users\Kashif\projectreportproduct
```

### Step 2: Create a Virtual Environment
```powershell
py -m venv venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

> **PowerShell Execution Policy Tip**:  
> If PowerShell displays `cannot be loaded because running scripts is disabled on this system`, permit script execution for your current session only:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

### Step 4: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 5: Run Exploratory Data Analysis (EDA)
Generates summary statistics and outputs high-resolution charts into `eda_outputs/`:
```powershell
python eda.py
```

### Step 6: Train Regression Model (Component 1)
Evaluates candidate regressors via cross-validation and serializes the champion model:
```powershell
python train.py
```

### Step 7: Train Classification Model (Component 2)
Trains the stratified Random Forest price-tier classifier:
```powershell
python classification.py
```

### Step 8: Train Clustering Model (Component 3)
Evaluates silhouette scores and segments the market:
```powershell
python clustering.py
```

### Step 9: Run Automated Tests
```powershell
python -m pytest
```

### Step 10: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
Streamlit will launch locally in your default web browser at `http://localhost:8501`.

---

## 10. Automated Tests & Verification

The automated test suite (`tests/test_models.py`) contains **11 unit and integration tests**:

1. `test_data_loading`: Verifies dataset exists, contains >20,000 rows, and validates non-negative prices.
2. `test_missing_data_error_handling`: Checks proper raising of `FileNotFoundError`.
3. `test_regression_model_prediction`: Validates that the loaded model outputs positive, finite dollar prices.
4. `test_regression_missing_features_error`: Confirms pipeline rejects incomplete input matrices.
5. `test_classification_model_prediction`: Verifies valid predicted category and probability distribution summing to 1.0.
6. `test_price_category_thresholds`: Tests boundary edge conditions for Budget, Mid-Range, and Premium labels.
7. `test_clustering_model_prediction`: Validates that cluster indices fall within valid range $[0, K-1]$.
8. `test_feature_schema_consistency`: Asserts that numerical and categorical schemas match pipeline transformers.
9. `test_investment_analyzer_discount`: Verifies negative delta label ("Below model estimate").
10. `test_investment_analyzer_fair_value`: Verifies tolerance band match ("Near model estimate").
11. `test_investment_analyzer_premium`: Verifies positive delta label ("Above model estimate").

Run tests anytime with:
```powershell
python -m pytest
```

---

## 11. Troubleshooting Common Errors

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError: 'kc_house_data.csv'` | Missing dataset file in `data/` | Place `kc_house_data.csv` inside `c:\Users\Kashif\projectreportproduct\data\`. |
| `Set-ExecutionPolicy` error on venv activate | Windows PowerShell script restriction | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your terminal. |
| Model files missing when launching app | Training scripts not run yet | Execute `python train.py`, `python classification.py`, and `python clustering.py`. |
| Port 8501 already in use | Previous Streamlit instance still running | Run `streamlit run app.py --server.port 8502` to use an alternate port. |
| Memory warning during RF fit | Too many parallel threads on laptop | Handled automatically in our scripts using `n_jobs=1`. |

---

## 12. Viva Preparation: 15 Questions & Answers

### Q1: What is the main objective of EstateIQ?
**Answer**: EstateIQ demonstrates an end-to-end Machine Learning pipeline applied to real estate intelligence. It combines exploratory data analysis, continuous valuation using regression, price tier categorization using classification, market grouping using clustering, and deal assessment in an interactive dashboard.

### Q2: What dataset did you use and what does it cover?
**Answer**: We used the King County House Sales dataset (`kc_house_data.csv`), containing 21,613 authentic residential transactions in King County, Washington (Seattle area) between May 2014 and May 2015.

### Q3: What is Target Leakage and how did you prevent it?
**Answer**: Target leakage happens when features containing knowledge of the target variable are present in the training input. We prevented this by removing `price` and all price-derived categories from feature matrix $X$, dropping `id` and `date`, and fitting all scalers and imputers solely on the training split inside a `Pipeline`.

### Q4: Why did Random Forest outperform Linear Regression?
**Answer**: Linear Regression assumes additive linear relationships ($y = \sum w_i x_i$). Real estate prices exhibit strong non-linear interactions: for example, building grade and waterfront views exponentially amplify price per square foot in luxury properties. Random Forest captures non-linear decision boundaries through recursive feature splitting.

### Q5: What does an $R^2$ score of 0.8529 mean?
**Answer**: The Coefficient of Determination ($R^2$) measures the fraction of variance in house prices explained by the model. An $R^2$ of 0.8529 indicates that 85.29% of the variability in King County property prices is explained by our feature set, while ~14.7% is due to unobserved factors like interior styling or buyer urgency.

### Q6: What is the difference between MAE and RMSE?
**Answer**: **Mean Absolute Error (MAE)** calculates the average absolute dollar error ($\frac{1}{n} \sum |y_i - \hat{y}_i|$), penalizing all errors linearly. **Root Mean Squared Error (RMSE)** squares errors before averaging ($\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$), heavily penalizing large outliers. Because luxury home errors are large in dollar terms, our RMSE (\$149,108) is higher than our MAE (\$74,243).

### Q7: Why use K-Fold Cross-Validation rather than just a train/test split?
**Answer**: A single train/test split can be biased by lucky or unlucky test partitioning. K-Fold Cross-Validation splits the training data into $K$ folds, training on $K-1$ and validating on the remaining fold repeatedly. This ensures our model selection is robust and generalizable.

### Q8: Why did you use Stratified train/test splitting for classification?
**Answer**: In price classification, the three classes (Budget, Mid-Range, Premium) have unequal proportions (37%, 39%, 24%). Stratification guarantees that the training split and holdout test split preserve the exact same class ratio as the original dataset.

### Q9: What are Precision, Recall, and F1-Score?
**Answer**: 
- **Precision**: Of all properties predicted as Premium, what percentage were truly Premium? ($\frac{TP}{TP + FP}$).
- **Recall**: Of all actual Premium properties, what percentage did the model correctly identify? ($\frac{TP}{TP + FN}$).
- **F1-Score**: The harmonic mean of Precision and Recall ($2 \times \frac{P \times R}{P + R}$), providing a balanced score when classes have varying costs of misclassification.

### Q10: How did you handle categorical variables like `zipcode`?
**Answer**: We used `OneHotEncoder(handle_unknown="ignore")` inside a `ColumnTransformer`. One-Hot Encoding creates separate binary indicator columns for each zipcode. This avoids treating zipcodes as ordinal numbers (which would falsely imply that zipcode 98199 is "greater than" 98001).

### Q11: What is K-Means clustering and how does it work?
**Answer**: K-Means is an unsupervised clustering algorithm that groups unlabelled data points into $K$ distinct clusters. It initializes $K$ centroids, assigns each property to the nearest centroid using Euclidean distance, and iteratively updates centroids to the mean of their assigned members until convergence.

### Q12: How did you select the number of clusters $K$?
**Answer**: We evaluated candidate $K$ values from 2 to 6 using **Inertia** (within-cluster sum of squares) and the **Silhouette Score**. While $K=2$ produces a high mathematical silhouette, it merely divides homes into cheap vs expensive. We selected **$K=3$** because it creates three actionable real estate market personas: Budget & Starter Homes, Mid-Market Suburban Residences, and Executive & Luxury Estates.

### Q13: What does the Silhouette Score measure?
**Answer**: The Silhouette Score evaluates how well-separated clusters are on a scale of -1 to +1. For each sample $i$, it computes $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$, where $a(i)$ is the mean distance to points in the same cluster (cohesion) and $b(i)$ is the mean distance to points in the nearest neighboring cluster (separation).

### Q14: How does the Investment Analyzer work?
**Answer**: The user supplies an asking price and property specs. The model predicts an estimated reference fair value. The analyzer calculates the dollar spread ($\text{Asking} - \text{Estimated}$) and percentage spread. If the asking price is more than 10% below the estimate, it flags the home as a potential value opportunity.

### Q15: Can EstateIQ be used to value homes in India?
**Answer**: **No, not directly.** The model was trained on King County (Seattle, Washington) data in USD and square feet under US municipal building codes. Real estate values in Indian cities (e.g., Delhi, Mumbai, Bengaluru) depend on different economic parameters (rupees, carpet area vs super built-up area, localized infrastructure, municipal circle rates). To adapt it for India, one must retrain the pipeline on localized Indian property transaction data.

---

## 13. License & Ethical AI Usage

EstateIQ is developed for educational purposes under the MIT License. Valuations are machine learning approximations based on historical data and do not substitute for formal structural surveys, environmental inspections, or licensed real estate appraisals.

