# Training data and provenance

The original study used **105 manually collected sold-property records**, 35 each from Bondi Beach, Parramatta and Blacktown. Each original row included a public sold-listing source page. The referenced source file was `sydney_housing_sold_105.csv`, but it **is not published in this repository**. It was collected from realestate.com.au and redistribution rights have not been established. The notebook/report retain the original historical evidence; a fresh training run requires a copy of this CSV provided by the project owner with permission to use it.

Expected columns for training: `suburb`, `property_type`, `bedrooms`, `bathrooms`, `parking`, `land_size_m2`, `sale_date`, `sale_price`. Dates must be `YYYY-MM-DD`, and price is in AUD. Blank numeric values for `parking` and `land_size_m2` are permitted. The input to the *prediction UI* uses all of these columns except `sale_price`.

The committed `example_input.csv` is **illustrative inference input**: it contains no real sales, addresses, or labels and must **not** be used to claim any model evaluation results.

To reproduce training after obtaining the authorised original CSV:

```bash
python -m housing_ml.train --data data/sydney_housing_sold_105.csv
```

This generates `best_model.joblib` and `results/new_training_metrics.json` locally. Neither is checked in. Saved historical results in the notebook should not be treated as independently reproduced by this package.
