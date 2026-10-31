"""Central configuration for the Real Estate Investment Advisor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
EDA_DIR = ROOT / "eda_outputs"

RAW_CSV = DATA_DIR / "raw_india_housing_prices.csv"
CLEAN_CSV = DATA_DIR / "cleaned_india_housing_prices.csv"
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "Real_Estate_Investment_Advisor"

RANDOM_STATE = 42
HORIZON_YEARS = 5
BASE_GROWTH_RATE = 0.08          # fixed-rate baseline from the brief (r = 8%)
LAKH = 100_000                   # 1 lakh = 100,000 INR

# --- Growth-rate model for the 5-year target (see docs/METHODOLOGY.md) ---
# Final r = city rate + property-type adjustment + feature adjustments, clipped to [4%, 14%].
CITY_TIER_GROWTH = {
    "Mumbai": 0.095, "Delhi": 0.092, "New Delhi": 0.092, "Bangalore": 0.095, "Bengaluru": 0.095,
    "Hyderabad": 0.093, "Chennai": 0.088, "Pune": 0.090, "Kolkata": 0.082,
    "Ahmedabad": 0.085, "Gurgaon": 0.092, "Noida": 0.088,
}
DEFAULT_CITY_GROWTH = 0.075
PROPERTY_TYPE_ADJ = {"Apartment": 0.000, "Independent House": 0.003, "Villa": 0.006}

# --- Good-investment multi-factor score ---
GOOD_SCORE_THRESHOLD = 4         # score out of 7

# Features given to the models. Columns that define the labels directly
# (Investment_Score, Growth_Rate, City_*_Median_*) are deliberately excluded.
NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property",
    "Floor_No", "Total_Floors", "Floor_Ratio", "Nearby_Schools", "Nearby_Hospitals",
    "School_Density_Score", "Infrastructure_Score", "Amenity_Count", "Transport_Score",
    "Price_per_BHK", "PPSF_vs_City_Median", "PPSF_vs_CityType_Median", "Price_vs_City_Median",
]
BINARY_FEATURES = ["Parking_Space", "Security"]
CATEGORICAL_FEATURES = [
    "State", "City", "Property_Type", "Furnished_Status", "Facing", "Owner_Type",
    "Availability_Status", "Public_Transport_Accessibility",
]
