# Real Estate Investment Advisor: Methodology & Findings

## 1. Objective
Help an investor decide (a) whether a property is a **Good Investment** (classification) and (b) what it will be **worth in 5 years** (regression).

## 2. Data
`india_housing_prices.csv`: 250,000 listings, 23 columns, 20 states, 42 cities, 500 localities. No missing values and no duplicates were found.

### Preprocessing
| Issue | Action |
|---|---|
| Missing values | Median (numeric) / mode (categorical) imputation built into the pipeline (none triggered on this file) |
| Duplicates | Dropped on all columns except `ID` (none found) |
| `Floor_No` > `Total_Floors` in **46.5%** of rows | Logically impossible. `Total_Floors` raised to `Floor_No`; flagged in `Floor_Data_Fixed` instead of dropping half the data |
| `Price_per_SqFt` rounded to 2 dp in lakhs/sq ft | Recomputed in ₹/sq ft = Price × 100,000 / Size |
| Outliers | IQR flags for price/sq ft, size, price. Retained (7.9% of rows flagged on price/sq ft) |
| Scaling / encoding | `StandardScaler` on numeric features; one-hot encoding (rare levels grouped) on categoricals; fitted inside the scikit-learn pipeline |

### Engineered features
Price per sq ft (₹), **School Density Score** (0-10, weighted schools + hospitals), **Infrastructure Score** (0-10: schools, hospitals, transport, parking, security), Amenity Count and per-amenity flags, Transport Score, Floor Ratio, Price per BHK, and market-relative value ratios (price/sq ft vs city and city-and-type median, price vs city median).

## 3. Targets (rule-based, as the brief specifies)
**Future price (regression):** `Future = Current × (1 + r)^5`, where `r` is built per property from the city (metros ≈ 9-9.5%, others 7.5%), property type (Villa +0.6 pt, House +0.3 pt), infrastructure, transport access, under-construction status, age and amenities, clipped to 4-14%. Resulting `r` ranges 6.9-10.8% (mean 8.2%).
The fixed 8% rate from the brief is the baseline.

**Good Investment (classification):** a 7-point score, label = 1 when score ≥ 4. One point each for:
price/sq ft ≤ city-and-type median; price ≤ city median; BHK ≥ 3; ready-to-move; infrastructure ≥ median; projected growth ≥ median; ≥ 3 amenities.
Result: 55.6% of properties are labelled good.

## 4. Modelling and evaluation (revised leakage-aware workflow)
The training workflow uses four disjoint partitions: **60% fit / 15% validation / 10% calibration / 15% independent final test**. Market reference medians and target thresholds are calculated from the fit partition only, then reused unchanged for validation, calibration, final test and app inference. Unknown cities use fit-partition global medians. Model selection uses validation results; the selected classifier is probability-calibrated on the separate calibration partition using sigmoid (Platt) calibration. Final metrics are calculated on the untouched final test partition.

Classification metrics include accuracy, precision, recall, F1, ROC-AUC, Brier score and log loss. Regression metrics include RMSE, MAE and R², with the fixed 8% growth formula retained as a baseline. The app's existing model filenames, feature-importance CSVs, metrics JSON and reference-table JSON are preserved. MLflow logs and registers the selected models.

**Interpretation warning:** The labels and five-year prices are generated from rules in this project, not observed investment outcomes or historical resale transactions. Even an independent test score only measures how well a model recovers those generated rules. Calibration improves probability reliability against these generated labels; it does not make the result a real-world probability of investment success.

## 5. EDA findings (20 questions; charts in `eda_outputs/`)
* **Prices and sizes are near-uniform**: 10-500 lakhs and 500-5,000 sq ft, with no long right tail.
* **Size and price are uncorrelated** (r = -0.003). Price per sq ft is therefore driven by size by arithmetic, not by the market.
* **Location barely matters**: city mean prices span only 251-258 lakhs; state mean ₹/sq ft spans roughly 12,900-13,300.
* **Property type, furnishing, facing, parking, amenities, schools and hospitals** show no meaningful relationship with price or price per sq ft (differences under about 1%).
* BHK is evenly split (~20% each) in every city; owner type and availability status are each split about evenly.
* **Public transport** has no price effect, but the share of good investments rises from 42% (Low) to 69% (High) because transport feeds the growth-rate and thus the label.
* The brief mentions **crime rate**, but no such column exists; Security and Infrastructure Score are used as proxies.
* The data has **no transaction dates**, so "price trends" for top localities are shown by construction year.

## 6. Limitations (read before interpreting results)
1. **Targets are synthetic rules, not observed outcomes.** The models learn to reproduce the rules; the near-perfect scores show the pipeline works, not that it predicts real market returns. A real deployment needs historical resale prices.
2. **The dataset appears synthetic**: raw attributes are statistically independent of each other, unlike real housing data.
3. Growth rates are assumptions and should be re-based on real city-level appreciation data.
4. The revised training script calculates market reference medians and label thresholds on the fit partition only. The underlying dataset still has no real transaction history, so external validity remains limited.
5. Nothing here is financial advice.

## 7. Use cases
Retail buyers screening listings; investors comparing cities and property types; advisors producing scenario forecasts; analysts exploring market structure by state, city and BHK.
