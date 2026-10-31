"""Cleaning + feature engineering + rule-based target creation."""
import numpy as np
import pandas as pd

from . import config as C

AMENITY_LIST = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]
TRANSPORT_MAP = {"Low": 1, "Medium": 2, "High": 3}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values, duplicates, text normalisation, logical fixes, outlier flags."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    n0 = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"]).reset_index(drop=True)

    # Imputation (the supplied file has no nulls, but the pipeline stays robust)
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].fillna(df[c].mode().iloc[0]).astype(str).str.strip()

    # Standardise labels
    df["Furnished_Status"] = df["Furnished_Status"].replace({"Furnished": "Fully-furnished"})
    df["Owner_Type"] = df["Owner_Type"].replace({"Owner": "Individual"})

    # Logical fix: a floor number cannot exceed building height. ~46% of raw rows violate this,
    # so we raise Total_Floors to Floor_No rather than discard half the data.
    df["Floor_Data_Fixed"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)
    df["Total_Floors"] = np.maximum(df["Total_Floors"], df["Floor_No"]).clip(lower=1)

    # Raw Price_per_SqFt is rounded to 2 decimals in lakh/sqft (precision lost); recompute in INR
    df["Price_per_SqFt_INR"] = df["Price_in_Lakhs"] * C.LAKH / df["Size_in_SqFt"]

    # IQR outlier flags (kept, not removed; reported in EDA)
    for col in ["Price_per_SqFt_INR", "Size_in_SqFt", "Price_in_Lakhs"]:
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        df[f"{col}_Outlier"] = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).astype(int)
    df.attrs["dropped_duplicates"] = n0 - len(df)
    return df


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computable from a single property (used by both training and the app)."""
    df = df.copy()
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map(TRANSPORT_MAP)
    df["Amenity_Count"] = df["Amenities"].apply(lambda s: sum(a in s for a in AMENITY_LIST))
    for a in AMENITY_LIST:
        df[f"Has_{a}"] = df["Amenities"].str.contains(a).astype(int)
    if df["Parking_Space"].dtype == object or str(df["Parking_Space"].dtype).startswith(("str", "string")):
        df["Parking_Space"] = (df["Parking_Space"] == "Yes").astype(int)
        df["Security"] = (df["Security"] == "Yes").astype(int)
    df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"]
    df["Price_per_BHK"] = df["Price_in_Lakhs"] / df["BHK"]
    # School Density Score (0-10): weighted schools + hospitals access in the neighbourhood
    df["School_Density_Score"] = (df["Nearby_Schools"] / 10 * 7 + df["Nearby_Hospitals"] / 10 * 3).round(2)
    # Infrastructure score (0-10): schools, hospitals, transport, parking, security
    df["Infrastructure_Score"] = (
        0.30 * df["Nearby_Schools"] + 0.25 * df["Nearby_Hospitals"]
        + 0.25 * (df["Transport_Score"] / 3 * 10) + 0.10 * df["Parking_Space"] * 10
        + 0.10 * df["Security"] * 10
    ).round(2)
    return df


def build_reference_tables(df: pd.DataFrame) -> dict:
    """Market reference statistics used for relative-value features (saved for the app)."""
    return {
        "city_ppsf": df.groupby("City")["Price_per_SqFt_INR"].median().to_dict(),
        "city_type_ppsf": {f"{c}|{t}": v for (c, t), v in
                           df.groupby(["City", "Property_Type"])["Price_per_SqFt_INR"].median().items()},
        "city_price": df.groupby("City")["Price_in_Lakhs"].median().to_dict(),
        "infra_median": float(df["Infrastructure_Score"].median()),
        "global_ppsf": float(df["Price_per_SqFt_INR"].median()),
        "global_price": float(df["Price_in_Lakhs"].median()),
    }


def add_market_features(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    df = df.copy()
    global_ppsf = float(ref.get("global_ppsf", np.median(list(ref["city_ppsf"].values()))))
    global_price = float(ref.get("global_price", np.median(list(ref["city_price"].values()))))
    df["City_Median_PPSF"] = df["City"].map(ref["city_ppsf"]).fillna(global_ppsf)
    key = df["City"] + "|" + df["Property_Type"]
    df["City_Type_Median_PPSF"] = key.map(ref["city_type_ppsf"]).fillna(df["City_Median_PPSF"])
    df["City_Median_Price"] = df["City"].map(ref["city_price"]).fillna(global_price)
    df["PPSF_vs_City_Median"] = df["Price_per_SqFt_INR"] / df["City_Median_PPSF"]
    df["PPSF_vs_CityType_Median"] = df["Price_per_SqFt_INR"] / df["City_Type_Median_PPSF"]
    df["Price_vs_City_Median"] = df["Price_in_Lakhs"] / df["City_Median_Price"]
    return df


def growth_rate(df: pd.DataFrame) -> pd.Series:
    """Annual appreciation: city rate + property type + feature-based adjustments."""
    r = df["City"].map(C.CITY_TIER_GROWTH).fillna(C.DEFAULT_CITY_GROWTH)
    r = r + df["Property_Type"].map(C.PROPERTY_TYPE_ADJ).fillna(0)
    r = r + 0.0020 * (df["Infrastructure_Score"] - 5) / 5
    r = r + 0.0030 * (df["Transport_Score"] - 2)
    r = r + 0.0020 * (df["Availability_Status"] == "Under_Construction")
    r = r - 0.0010 * (df["Age_of_Property"] / 35)
    r = r + 0.0010 * (df["Amenity_Count"] - 2.5) / 2.5
    return r.clip(0.04, 0.14)


def make_targets(df: pd.DataFrame, growth_median: float | None = None, infra_median: float | None = None) -> pd.DataFrame:
    """Regression target (future price) and classification label (Good_Investment)."""
    df = df.copy()
    df["Growth_Rate"] = growth_rate(df)
    df["Future_Price_5Y_Lakhs"] = df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** C.HORIZON_YEARS
    gm = df["Growth_Rate"].median() if growth_median is None else growth_median
    infra_med = df["Infrastructure_Score"].median() if infra_median is None else infra_median

    # Multi-factor investment score (0-7): value + quality + readiness
    s = (df["Price_per_SqFt_INR"] <= df["City_Type_Median_PPSF"]).astype(int)   # cheap vs peers per sqft
    s += (df["Price_in_Lakhs"] <= df["City_Median_Price"]).astype(int)          # price <= city median
    s += (df["BHK"] >= 3).astype(int)
    s += (df["Availability_Status"] == "Ready_to_Move").astype(int)
    s += (df["Infrastructure_Score"] >= infra_med).astype(int)
    s += (df["Growth_Rate"] >= gm).astype(int)
    s += (df["Amenity_Count"] >= 3).astype(int)
    df["Investment_Score"] = s
    df["Good_Investment"] = (s >= C.GOOD_SCORE_THRESHOLD).astype(int)
    return df


def build_clean_dataset(raw_path=C.RAW_CSV):
    df = clean(pd.read_csv(raw_path))
    df = add_row_features(df)
    ref = build_reference_tables(df)
    ref["growth_median"] = float(growth_rate(df).median())
    df = add_market_features(df, ref)
    df = make_targets(df, ref["growth_median"])
    return df, ref
