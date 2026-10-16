# Historical results and new-run outputs

The **original submitted notebook** reported a cross-validation-selected Random Forest with held-out **MAE A$509,178** and **R² 0.803** on a 21-property test set. Its training CV MAE was approximately A$471,511. These were **saved notebook results**, not re-executed as part of this repository refactor.

New runs through `python -m housing_ml.train` save separate metrics to `results/new_training_metrics.json` by default. The new training script selects model and hyperparameters entirely using training-fold CV and only then evaluates the selected winner on the held-out test partition. Its new results may differ from the original notebook and must be reported separately.

Because the dataset is small and manually selected, error on high-value or unusual properties can be substantial. This is a learning/demo project, not a tool for valuation or financial decisions.
