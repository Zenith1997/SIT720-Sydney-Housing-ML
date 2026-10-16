# Sydney Housing Price Prediction

Compare regression models and turn a saved prediction pipeline into a decision-support workflow.

**Python · pandas · scikit-learn · Random Forest · Gradio**

## Project at a glance

Random Forest was selected using training cross-validation MAE (A$471,511). Its held-out MAE was A$509,178 and R² was 0.803. Other candidates performed differently on the held-out metrics; these numbers do not establish universal superiority.

## Technical work

- Feature engineering and preprocessing pipelines
- Ridge, Gradient Boosting and Random Forest comparison
- Cross-validation-based model selection
- Residual and error analysis; saved-pipeline and Gradio workflow

## Explore the evidence

- [SIT720_8_1D_Zenith_Anthony_feedback_fixed.ipynb](SIT720_8_1D_Zenith_Anthony_feedback_fixed.ipynb)
- [SIT720_8_1D_Report_Zenith_Anthony_feedback_fixed.pdf](SIT720_8_1D_Report_Zenith_Anthony_feedback_fixed.pdf)

## Run locally

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter notebook
```

Use the notebook's DATA setting. The source workflow expects sydney_housing_sold_105.csv and additional comparison CSVs; these support files are not currently bundled in this repository.

Open the linked notebook and run the cells in order. Package names reflect the source imports; a fresh environment was not tested during this documentation update. Dataset downloads require internet access.

## Limitations and status

105 sold-property records across Bondi Beach, Parramatta and Blacktown; small sample, limited geographic coverage and high absolute prediction error.

The repository preserves saved notebook outputs. Results are from those outputs, not a new execution.

**Author:** Zenith Anthony Dabarera · SIT720 Machine Learning  
[AI/ML portfolio](https://github.com/Zenith1997)
