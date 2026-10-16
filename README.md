# Sydney Housing Price Estimation

A small-data machine-learning study with a reusable Python pipeline and Gradio interface.

**Python | scikit-learn | pandas | Random Forest | Gradient Boosting | Gradio**

## What I learned

This project compares three regression approaches on 105 manually collected sales from Bondi Beach, Parramatta and Blacktown. The more interesting challenge was understanding why unusual, expensive houses were difficult to estimate from basic tabular features.

The original saved notebook reports a **Random Forest**, selected by training cross-validation, with test **MAE A$509,178** and **R² 0.803** on 21 held-out properties. These are historical notebook results, not results independently re-run for this GitHub upgrade.

## Browse the code

- [Original executed notebook](SIT720_8_1D_Zenith_Anthony_feedback_fixed.ipynb) and [project report](SIT720_8_1D_Report_Zenith_Anthony_feedback_fixed.pdf)
- [Training script](src/housing_ml/train.py), [feature preparation](src/housing_ml/features.py), and [prediction code](src/housing_ml/modeling.py)
- [Gradio user interface](app.py) for a single property or a batch CSV
- [Example input](data/example_input.csv), [data provenance](data/README.md) and [historical results](results/README.md)
- [Automated tests](tests/) and [GitHub Actions workflow](.github/workflows/tests.yml)

## Run it

Python 3.10+ is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m housing_ml.train --data data/sydney_housing_sold_105.csv
python app.py
```

**Data limitation:** The original source CSV is not included publicly. Training needs the owner's original 105-property CSV and permission to use it. Without that CSV, the new training command and app cannot make real predictions. See [data instructions](data/README.md). The sample input is illustrative, not measured sales.

Run tests with `python -m pip install -r requirements-dev.txt` and `python -m pytest -q`.

This is an educational decision-support model, **not a property valuation service**. The input omits property condition, views, precise street position and other major price drivers. The original dataset is small and not representative of all Sydney property sales.

**Preserved source:** The submitted notebook and report remain unchanged.