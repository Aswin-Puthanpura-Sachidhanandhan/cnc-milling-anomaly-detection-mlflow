# CNC Milling Anomaly Detection with MLflow and Databricks

## Project Overview

This project demonstrates a complete production-oriented machine learning lifecycle for CNC milling anomaly detection.

The workflow includes:

- data preparation and feature engineering
- leakage-safe group-based train/test splitting
- comparison of multiple classical and ensemble-learning models
- MLflow experiment tracking
- model registration and versioning
- Champion–Challenger evaluation
- automatic promotion and rejection rules
- automatic Databricks serving endpoint updates
- final real-time inference validation

## Machine Learning Problem

The objective is to classify each CNC machining segment as:

- `0` — Normal machining operation
- `1` — Anomalous machining operation

## Dataset

Dataset: **Multi-Sensor Data for Metal Milling Anomaly Detection**

Kaggle source:

https://www.kaggle.com/datasets/manufuturetoday/multi-sensor-for-metal-milling-anomaly

The project uses 24 complete MTC-label experiment pairs. The large raw accelerometer files were excluded from the final pipeline to keep the project computationally manageable.

## Refined Dataset

- Rows: 1,107 machining segments
- Columns: 197
- Normal observations: 814
- Anomalous observations: 293
- Delta table: `default.refined_mtc_dataset`

## Data Splitting Strategy

A group-based split was used with the experiment folder as the grouping variable.

This ensures that observations from the same machining experiment cannot appear in both the training and testing sets.

- Training rows: 942
- Testing rows: 165
- Training experiments: 19
- Testing experiments: 5
- Experiment overlap: none

## Model Comparison

| Model | F1 Score | Recall | Precision | ROC-AUC |
|---|---:|---:|---:|---:|
| Decision Tree | 0.8867 | 0.9375 | 0.8411 | 0.8463 |
| Extra Trees | 0.8515 | 0.8958 | 0.8113 | 0.9404 |
| Gradient Boosting | 0.8505 | 0.9479 | 0.7712 | 0.9390 |
| XGBoost | 0.8311 | 0.9479 | 0.7398 | 0.8915 |
| Random Forest | 0.8224 | 0.9167 | 0.7458 | 0.9004 |

## Champion–Challenger Promotion Rule

A challenger is promoted only when:

1. its F1-score improves by at least `0.01`, and
2. its recall does not decrease by more than `0.02`.

```python
promote = (
    challenger_f1 >= champion_f1 + 0.01
    and challenger_recall >= champion_recall - 0.02
)
```

## Production Lifecycle

### Version 1 — Random Forest

Random Forest was selected as the initial production model.

- F1-score: 0.8224
- Recall: 0.9167
- Registry version: 1
- Result: Registered and deployed

### XGBoost Challenger

XGBoost improved recall but did not achieve the required minimum F1-score gain.

- F1 gain: 0.0087
- Result: Rejected
- Registry action: Not registered
- Production model remained Version 1

### Version 2 — Gradient Boosting

Gradient Boosting passed both quality gates.

- F1-score: 0.8505
- Recall: 0.9479
- Registry version: 2
- Result: Automatically promoted and deployed

### Extra Trees Challenger

Extra Trees achieved only a very small F1 improvement and reduced recall beyond the accepted tolerance.

- Result: Rejected
- Registry action: Not registered
- Production model remained Version 2

### Version 3 — Decision Tree

The Decision Tree achieved the strongest F1-score while maintaining acceptable anomaly recall.

- Accuracy: 0.8606
- Precision: 0.8411
- Recall: 0.9375
- F1-score: 0.8867
- ROC-AUC: 0.8463
- Registry version: 3
- Result: Automatically promoted as the final Champion

## Final Production State

- Registered model: `workspace.default.cnc_milling_anomaly_final`
- Champion alias: Version 3
- Serving endpoint: `cnc-milling-final-endpoint`
- Endpoint version: 3
- Traffic allocation: 100%
- Endpoint status: Ready
- Champion and endpoint synchronized: True

## Final Inference Validation

The final Version 3 endpoint was tested with known samples.

| Sample | Actual Label | Local Prediction | Endpoint Prediction | Result |
|---|---:|---:|---:|---|
| Normal machining segment | 0 | 0 | 0 | Correct |
| Anomalous machining segment | 1 | 1 | 1 | Correct |

This confirms that the deployed endpoint returns predictions consistent with the locally evaluated Champion model.

## MLflow and Databricks Components

The project uses:

- MLflow experiment tracking
- MLflow model logging
- Unity Catalog Model Registry
- registered model aliases
- Champion–Challenger promotion logic
- Databricks Model Serving
- endpoint configuration updates
- real-time API inference

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_clean_model_lifecycle.ipynb
└── screenshots/
    ├── 01_dataset_file_summary.png
    ├── 02_refined_dataset_summary.png
    ├── 03_group_split_no_leakage.png
    ├── 04_model_ranking.png
    ├── 05_version1_random_forest_deployed.png
    ├── 06_xgboost_rejected.png
    ├── 07_gradient_boosting_promoted_v2.png
    ├── 08_extra_trees_rejected.png
    ├── 09_decision_tree_promoted_v3.png
    ├── 10_final_normal_inference.png
    ├── 11_final_anomaly_inference.png
    ├── 12_complete_ml_lifecycle.png
    ├── 13_registered_model_versions.png
    └── 14_final_endpoint_version3.png
```

## Notebook Execution Order

1. `01_data_preparation.ipynb`
2. `02_clean_model_lifecycle.ipynb`

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- MLflow
- Databricks
- Delta Lake
- Unity Catalog
- Databricks Model Serving

## Limitations

- Only 24 complete MTC-label experiments were available.
- The dataset is moderately imbalanced.
- Raw accelerometer data was excluded.
- Model performance may vary on unseen machining conditions.
- Multiple models were compared on the same holdout set, so a larger independent validation set would strengthen future evaluation.

## Future Improvements

- add accelerometer time-domain and frequency-domain features
- collect additional labeled machining experiments
- use cost-sensitive anomaly evaluation
- add production drift monitoring
- automate retraining on newly collected data
- evaluate the final model on an independent external test set

## Final Conclusion

The project successfully demonstrates the complete required lifecycle:

```text
Train and evaluate multiple models
        ↓
Register and deploy a good initial model
        ↓
Evaluate challengers automatically
        ↓
Reject models that fail the quality gate
        ↓
Promote and deploy better models
        ↓
Synchronize Champion alias and endpoint
        ↓
Validate final real-time inference
```
