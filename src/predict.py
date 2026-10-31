"""Inference helpers: turn a user's property description into model inputs and predictions."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from . import config as C
from . import features as F
from .train import FEATURES


@lru_cache(maxsize=1)
def load_artifacts():
    clf = joblib.load(C.MODEL_DIR / "classification_model.joblib")
    reg = joblib.load(C.MODEL_DIR / "regression_model.joblib")
    ref = json.load(open(C.MODEL_DIR / "reference_tables.json"))
    metrics = json.load(open(C.MODEL_DIR / "metrics.json"))
    return clf, reg, ref, metrics


def build_input(p: dict, ref: dict) -> pd.DataFrame:
    """p: dict of raw user inputs -> single-row frame with all model features."""
    row = {
        "State": p["State"], "City": p["City"], "Property_Type": p["Property_Type"], "BHK": p["BHK"],
        "Size_in_SqFt": p["Size_in_SqFt"], "Price_in_Lakhs": p["Price_in_Lakhs"],
        "Age_of_Property": p["Age_of_Property"], "Furnished_Status": p["Furnished_Status"],
        "Floor_No": p["Floor_No"], "Total_Floors": max(p["Total_Floors"], p["Floor_No"], 1),
        "Nearby_Schools": p["Nearby_Schools"], "Nearby_Hospitals": p["Nearby_Hospitals"],
        "Public_Transport_Accessibility": p["Public_Transport_Accessibility"],
        "Parking_Space": "Yes" if p["Parking_Space"] else "No",
        "Security": "Yes" if p["Security"] else "No",
        "Amenities": ", ".join(p["Amenities"]), "Facing": p["Facing"],
        "Owner_Type": p["Owner_Type"], "Availability_Status": p["Availability_Status"],
    }
    df = pd.DataFrame([row])
    df["Price_per_SqFt_INR"] = df["Price_in_Lakhs"] * C.LAKH / df["Size_in_SqFt"]
    df = F.add_row_features(df)
    df = F.add_market_features(df, ref)
    return df


def predict(p: dict) -> dict:
    clf, reg, ref, _ = load_artifacts()
    df = build_input(p, ref)
    X = df[FEATURES]
    proba = float(clf.predict_proba(X)[0, 1])
    future = float(reg.predict(X)[0])
    price = float(df["Price_in_Lakhs"].iloc[0])
    years = C.HORIZON_YEARS
    cagr = (future / price) ** (1 / years) - 1
    return {
        "good_probability": proba, "is_good": proba >= 0.5,
        # This is the calibrated probability of the predicted class, not a guarantee.
        "confidence": max(proba, 1 - proba),
        "future_price": future, "cagr": cagr, "gain": future - price,
        "trajectory": [price * (1 + cagr) ** t for t in range(years + 1)],
        "fixed_8pct": price * (1 + C.BASE_GROWTH_RATE) ** years,
        "ppsf": float(df["Price_per_SqFt_INR"].iloc[0]),
        "city_ppsf": float(df["City_Median_PPSF"].iloc[0]),
        "ppsf_ratio": float(df["PPSF_vs_CityType_Median"].iloc[0]),
        "infra": float(df["Infrastructure_Score"].iloc[0]),
        "school_density": float(df["School_Density_Score"].iloc[0]),
        "row": df,
    }
