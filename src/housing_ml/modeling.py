"""Model bundle loading and consistent prediction for the Gradio frontend."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import MODEL_FEATURES, make_features


def load_bundle(path: str | Path = "best_model.joblib") -> dict:
    """Load a locally trained model. Never load untrusted joblib files."""
    model_file = Path(path)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_file}. First run the original notebook "
            "with the 105-row dataset or run `python -m housing_ml.train --data ...`."
        )
    bundle = joblib.load(model_file)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Expected a model bundle containing a fitted 'model' key")
    expected = list(MODEL_FEATURES)
    if list(bundle.get("features", [])) != expected:
        raise ValueError("Model feature schema does not match the project feature schema")
    return bundle


def predict_prices(inputs: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return a copy of raw inputs with model estimates in AUD.

    This is an educational decision-support estimate, not a valuation.
    """
    features = make_features(inputs)
    predictions = np.asarray(bundle["model"].predict(features), dtype=float)
    if predictions.shape != (len(inputs),) or not np.isfinite(predictions).all():
        raise ValueError("The model returned invalid predictions")
    output = inputs.copy()
    output["estimated_sale_price_aud"] = predictions.round(0).astype("int64")
    return output
