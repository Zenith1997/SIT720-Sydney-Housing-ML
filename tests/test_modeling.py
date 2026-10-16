from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from housing_ml.features import MODEL_FEATURES, make_features
from housing_ml.modeling import load_bundle, predict_prices


def example_input():
    return pd.DataFrame([{
        "suburb": "Blacktown", "property_type": "House", "bedrooms": 3,
        "bathrooms": 1, "parking": 1, "land_size_m2": 400,
        "sale_date": "2025-05-01",
    }])


def fitted_demo_bundle():
    """Dummy training verifies the plumbing, not the project's real MAE."""
    inp = example_input()
    model = DummyRegressor(strategy="constant", constant=750_000)
    model.fit(make_features(inp), np.array([750_000]))
    return {"model": model, "features": list(MODEL_FEATURES), "best_model_name": "test dummy"}


def test_predict_retains_inputs_and_adds_estimate():
    incoming = example_input()
    output = predict_prices(incoming, fitted_demo_bundle())
    assert output.loc[0, "estimated_sale_price_aud"] == 750_000
    assert "estimated_sale_price_aud" not in incoming.columns
    assert output.loc[0, "suburb"] == "Blacktown"


def test_bundle_round_trip(tmp_path: Path):
    path = tmp_path / "trained.joblib"
    joblib.dump(fitted_demo_bundle(), path)
    loaded = load_bundle(path)
    assert len(predict_prices(example_input(), loaded)) == 1


def test_missing_model_has_actionable_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="First run"):
        load_bundle(tmp_path / "missing.joblib")


def test_schema_mismatch_rejected(tmp_path: Path):
    path = tmp_path / "bad.joblib"
    bundle = fitted_demo_bundle()
    bundle["features"] = ["wrong"]
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="feature schema"):
        load_bundle(path)
