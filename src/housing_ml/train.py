"""Train and export a Sydney housing model with training-only model selection.

This is a standalone reproducible route. The original assignment notebook and
its historical results remain unchanged, and new metrics are saved separately.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES,
    RAW_FEATURES, make_features,
)


def train(data: Path, model_out: Path, metrics_out: Path, seed: int = 42) -> dict:
    if not data.is_file():
        raise FileNotFoundError(
            f"Training data not found: {data}. See data/README.md; "
            "the original manually collected records are not redistributable here."
        )
    df = pd.read_csv(data)
    if "sale_price" not in df:
        raise ValueError("Training data is missing the sale_price target")
    y = pd.to_numeric(df["sale_price"], errors="coerce")
    if y.isna().any() or y.le(0).any():
        raise ValueError("sale_price must contain positive AUD values")
    X = make_features(df.loc[:, RAW_FEATURES])
    if len(X) < 30 or X["suburb"].value_counts().min() < 5:
        raise ValueError("Too few records for a stratified 5-fold demonstration")

    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=X["suburb"], random_state=seed
    )
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_splits = list(cv.split(X_train, X_train["suburb"]))

    def preprocessor(scale: bool) -> ColumnTransformer:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            numeric_steps.append(("scale", StandardScaler()))
        return ColumnTransformer([
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
        ])

    candidates = {
        "Ridge": (Pipeline([("prep", preprocessor(True)), ("model", Ridge())]),
                  {"model__alpha": [0.1, 1, 10, 100, 300]}),
        "Random Forest": (Pipeline([("prep", preprocessor(False)),
                                    ("model", RandomForestRegressor(random_state=seed, n_jobs=1))]),
                          {"model__n_estimators": [100, 200],
                           "model__max_depth": [4, 6, None],
                           "model__min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (Pipeline([("prep", preprocessor(False)),
                                        ("model", GradientBoostingRegressor(random_state=seed))]),
                              {"model__n_estimators": [100, 180],
                               "model__learning_rate": [0.05, 0.1],
                               "model__max_depth": [2, 3],
                               "model__min_samples_leaf": [2]}),
    }

    best_name, best_search, best_cv_mae = None, None, float("inf")
    candidate_cv_mae = {}
    for name, (pipe, grid) in candidates.items():
        search = GridSearchCV(
            pipe, grid, scoring="neg_mean_absolute_error", cv=cv_splits,
            n_jobs=1, refit=True, error_score="raise",
        )
        search.fit(X_train, y_train)
        cv_mae = -float(search.best_score_)
        candidate_cv_mae[name] = cv_mae
        if cv_mae < best_cv_mae:
            best_name, best_search, best_cv_mae = name, search, cv_mae

    # The held-out test set is touched only after selecting the winning model.
    pred = best_search.best_estimator_.predict(X_test)
    metrics = {
        "data_rows": int(len(df)),
        "train_rows": int(len(idx_train)),
        "test_rows": int(len(idx_test)),
        "selected_model": best_name,
        "selection_method": "minimum training-only 5-fold CV MAE",
        "cv_mae_aud": candidate_cv_mae,
        "selected_hyperparameters": best_search.best_params_,
        "heldout_test_mae_aud": float(mean_absolute_error(y_test, pred)),
        "heldout_test_rmse_aud": float(np.sqrt(mean_squared_error(y_test, pred))),
        "heldout_test_r2": float(r2_score(y_test, pred)),
        "seed": seed,
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": best_search.best_estimator_,
        "features": list(MODEL_FEATURES),
        "best_model_name": best_name,
    }, model_out)
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sydney_housing_sold_105.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("best_model.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("results/new_training_metrics.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.model_out, args.metrics_out, args.seed), indent=2))


if __name__ == "__main__":
    main()
