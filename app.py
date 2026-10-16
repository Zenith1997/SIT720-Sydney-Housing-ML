"""Gradio single-property and batch CSV estimation app.

Requires a locally trained best_model.joblib. No model is bundled with this
repository. Do not load arbitrary third-party joblib files.
"""

import os
from pathlib import Path
import tempfile

import pandas as pd

from housing_ml import load_bundle, predict_prices

MODEL_PATH = Path(os.environ.get("HOUSING_MODEL_PATH", "best_model.joblib"))


def single_property(suburb, property_type, bedrooms, bathrooms, parking, land_size_m2, sale_date):
    raw = pd.DataFrame([{
        "suburb": suburb, "property_type": property_type,
        "bedrooms": bedrooms, "bathrooms": bathrooms,
        "parking": parking, "land_size_m2": land_size_m2,
        "sale_date": sale_date,
    }])
    result = predict_prices(raw, load_bundle(MODEL_PATH))
    price = int(result.loc[0, "estimated_sale_price_aud"])
    return (f"A${price:,.0f} (experimental estimate only — not a valuation; "
            "original held-out MAE was about A$509,178)")


def batch_file(csv_path):
    if csv_path is None:
        raise ValueError("Upload a CSV with the seven columns shown in data/example_input.csv")
    inputs = pd.read_csv(csv_path)
    result = predict_prices(inputs, load_bundle(MODEL_PATH))
    with tempfile.NamedTemporaryFile(mode="w", suffix="_housing_predictions.csv", delete=False, encoding="utf-8") as temp:
        result.to_csv(temp.name, index=False)
        return temp.name


def create_demo():
    import gradio as gr

    with gr.Blocks(title="Sydney Housing | Educational Price Estimates") as demo:
        gr.Markdown(
            "# Sydney Housing Price Estimator\n"
            "**Experimental coursework model, not financial or property valuation advice.** "
            "Trained on 105 manually collected sales in three suburbs; typical errors can be large. "
            "You must train/export `best_model.joblib` before predictions can run."
        )
        with gr.Tab("Single property"):
            with gr.Row():
                suburb = gr.Dropdown(["Bondi Beach", "Parramatta", "Blacktown"], label="Suburb", value="Parramatta")
                property_type = gr.Textbox(label="Property type (match training data)", value="House")
            with gr.Row():
                bedrooms = gr.Number(label="Bedrooms", value=3, precision=0)
                bathrooms = gr.Number(label="Bathrooms", value=2, precision=0)
                parking = gr.Number(label="Parking spaces (optional)", value=1, precision=0)
                land = gr.Number(label="Land area (m², optional)", value=350)
            sale_date = gr.Textbox(label="Reference sale date (YYYY-MM-DD)", value="2025-01-15")
            estimate = gr.Textbox(label="Estimated sale price (AUD)")
            gr.Button("Estimate price").click(
                single_property, [suburb, property_type, bedrooms, bathrooms, parking, land, sale_date], estimate
            )
        with gr.Tab("Batch CSV"):
            gr.Markdown("Upload a CSV matching `data/example_input.csv`; it must have all seven raw columns.")
            infile = gr.File(label="Properties CSV", file_types=[".csv"], type="filepath")
            outfile = gr.File(label="CSV with model estimates")
            gr.Button("Predict batch").click(batch_file, infile, outfile)
    return demo


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No trained model at {MODEL_PATH}. Run `python -m housing_ml.train --data "
            "data/sydney_housing_sold_105.csv` first. See data/README.md."
        )
    create_demo().launch()
