import pandas as pd
import pytest

from housing_ml.features import MODEL_FEATURES, make_features


def sample(**overrides):
    record = dict(
        suburb="Parramatta", property_type="House", bedrooms=3,
        bathrooms=2, parking=1, land_size_m2=360, sale_date="2025-07-12",
    )
    record.update(overrides)
    return pd.DataFrame([record])


def test_feature_names_and_engineering_match_original_notebook():
    x = make_features(sample())
    assert list(x.columns) == list(MODEL_FEATURES)
    assert x.loc[0, "sale_year"] == 2025
    assert x.loc[0, "sale_month"] == 7
    assert x.loc[0, "sale_quarter"] == "3"
    assert x.loc[0, "bed_bath_total"] == 5
    assert x.loc[0, "land_per_bed"] == 120
    assert x.loc[0, "parking_per_bed"] == pytest.approx(1 / 3)


def test_missing_parking_and_land_are_imputable():
    x = make_features(sample(parking=None, land_size_m2=None))
    assert pd.isna(x.loc[0, "parking_per_bed"])
    assert pd.isna(x.loc[0, "land_per_bed"])


def test_zero_bedrooms_does_not_divide_by_zero():
    x = make_features(sample(bedrooms=0))
    assert pd.isna(x.loc[0, "land_per_bed"])


@pytest.mark.parametrize("bad", ["17/10/2025", "tomorrow", "", "2026-02-30"])
def test_reject_invalid_dates(bad):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        make_features(sample(sale_date=bad))


def test_reject_unknown_suburb():
    with pytest.raises(ValueError, match="three|Unsupported"):
        make_features(sample(suburb="Melbourne"))


def test_reject_negative_or_fractional_rooms():
    with pytest.raises(ValueError):
        make_features(sample(bedrooms=-1))
    with pytest.raises(ValueError):
        make_features(sample(bathrooms=1.5))


def test_reject_missing_columns():
    with pytest.raises(ValueError, match="Missing required input columns"):
        make_features(sample().drop(columns="property_type"))
