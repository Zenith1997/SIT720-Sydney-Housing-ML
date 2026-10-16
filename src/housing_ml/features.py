"""Feature preparation shared between training and the inference UI.

This module intentionally mirrors the feature engineering in the original
published notebook; it does not invent new predictive features.
"""

import numpy as np
import pandas as pd

RAW_FEATURES = (
    "suburb", "property_type", "bedrooms", "bathrooms", "parking",
    "land_size_m2", "sale_date",
)
MODEL_FEATURES = (
    "suburb", "property_type", "bedrooms", "bathrooms", "parking",
    "land_size_m2", "sale_year", "sale_month", "sale_quarter",
    "bed_bath_total", "land_per_bed", "parking_per_bed",
)
NUMERIC_FEATURES = (
    "bedrooms", "bathrooms", "parking", "land_size_m2",
    "sale_year", "sale_month", "bed_bath_total", "land_per_bed", "parking_per_bed",
)
CATEGORICAL_FEATURES = ("suburb", "property_type", "sale_quarter")
KNOWN_SUBURBS = ("Bondi Beach", "Parramatta", "Blacktown")


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate property inputs and build the 12 original model features.

    Blank parking and land-size entries are allowed and imputed later by the
    fitted training pipeline. Invalid dates and negative sizes are rejected.
    Predictions outside the three training suburbs are intentionally blocked.
    """
    missing = sorted(set(RAW_FEATURES).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("No property rows were provided")

    x = frame.loc[:, RAW_FEATURES].copy()
    for col in ("suburb", "property_type"):
        x[col] = x[col].astype("string").str.strip()
        if x[col].isna().any() or x[col].eq("").any():
            raise ValueError(f"{col} cannot be blank")

    unknown = sorted(set(x["suburb"].unique()).difference(KNOWN_SUBURBS))
    if unknown:
        raise ValueError(
            "The model covers Bondi Beach, Parramatta and Blacktown only. "
            f"Unsupported suburb(s): {', '.join(unknown)}"
        )

    for col in ("bedrooms", "bathrooms", "parking", "land_size_m2"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
        if col in ("bedrooms", "bathrooms") and x[col].isna().any():
            raise ValueError(f"{col} must contain a valid number")
        if x[col].lt(0).any():
            raise ValueError(f"{col} cannot be negative")
    for col in ("bedrooms", "bathrooms", "parking"):
        valid = x[col].dropna()
        if not valid.eq(valid.round()).all():
            raise ValueError(f"{col} must contain whole numbers")

    # Explicit date format reduces day/month ambiguity in batch CSV uploads.
    dates = pd.to_datetime(x["sale_date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("sale_date must use YYYY-MM-DD for every row")
    x["sale_year"] = dates.dt.year
    x["sale_month"] = dates.dt.month
    x["sale_quarter"] = dates.dt.quarter.astype(str)
    x["bed_bath_total"] = x["bedrooms"] + x["bathrooms"]
    nonzero_beds = x["bedrooms"].replace(0, np.nan)
    x["land_per_bed"] = x["land_size_m2"] / nonzero_beds
    x["parking_per_bed"] = x["parking"] / nonzero_beds
    return x.loc[:, MODEL_FEATURES]
