# CNC Milling Anomaly Detection with Ensemble Learning and MLflow

## Project Overview

This project implements a production-style machine learning lifecycle for detecting anomalies in CNC metal-milling operations using industrial multi-sensor data.

The project includes data preparation, feature engineering, group-based splitting, ensemble model comparison, MLflow experiment tracking, model registration, Champion–Challenger evaluation, automatic promotion control, Databricks Model Serving, and real-time API inference.

## Machine Learning Problem

The model classifies each CNC machining segment as:

- `0` — Normal machining operation
- `1` — Anomalous machining operation

## Dataset

Dataset: **Multi-Sensor Data for Metal Milling Anomaly Detection**

Kaggle source:

https://www.kaggle.com/datasets/manufuturetoday/multi-sensor-for-metal-milling-anomaly

The project uses 24 complete MTC-label experiment pairs. The large accelerometer files were excluded from the baseline pipeline to keep the project computationally manageable.

## Refined Dataset

- Rows: 1,107
- Columns: 197
- Normal observations: 814
- Anomalous observations: 293
- Delta table: `default.refined_mtc_dataset`

## Data Splitting Strategy

A group-based split was used so the same machining experiment could not appear in both training and testing sets.

- Training rows: 942
- Test rows: 165
- Training experiments: 19
- Test experiments: 5
- Experiment overlap: none

## Baseline Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.8606 | 0.8411 | 0.9375 | 0.8867 | 0.8463 |
| Gradient Boosting | 0.8061 | 0.7712 | 0.9479 | 0.8505 | 0.9390 |
| Random Forest | 0.7455 | 0.7177 | 0.9271 | 0.8091 | 0.8966 |

The Decision Tree achieved the highest F1-score and was selected as the Champion.

## MLflow Tracking and Registry

MLflow experiment:

`/Shared/cnc_milling_anomaly_detection`

Registered model:

`cnc_milling_anomaly_model`

Registered versions:

- Version 1 — Decision Tree
- Version 2 — Tuned Random Forest

Version 1 uses the alias:

`Champion`

## Challenger Evaluation

| Model | F1 Score | Recall | Decision |
|---|---:|---:|---|
| Champion Decision Tree | 0.8867 | 0.9375 | Champion |
| Extra Trees | 0.8087 | 0.9688 | Rejected |
| XGBoost | 0.8190 | 0.9896 | Rejected |
| Weighted Voting Ensemble | 0.8400 | 1.0000 | Rejected |

Some challengers improved anomaly recall but produced more false alarms, reducing F1-score.

## Automatic Promotion Gate

```python
promote = (
    challenger_f1 > champion_f1
    and challenger_recall >= champion_recall - 0.02
)
```

The final challenger did not pass the gate, so Version 1 remained the Champion.

## Model Serving

Endpoint:

`cnc-milling-anomaly-endpoint`

Served model:

- Model: `cnc_milling_anomaly_model`
- Version: `1`
- Traffic: `100%`

Example response:

```json
{
  "predictions": [0]
}
```

Prediction mapping:

- `0` — Normal
- `1` — Anomaly

## Serving Validation

| Sample | Actual Label | Endpoint Prediction | Result |
|---|---:|---:|---|
| Known normal segment | 0 | 0 | Correct |
| Known anomaly segment | 1 | 1 | Correct |

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_mlflow_models.ipynb
│   ├── 03_improved_model.ipynb
│   └── 04_model_inference_and_serving.ipynb
└── screenshots/
    └── project evidence screenshots
```

## Notebook Execution Order

1. `01_data_preparation.ipynb`
2. `02_baseline_mlflow_models.ipynb`
3. `03_improved_model.ipynb`
4. `04_model_inference_and_serving.ipynb`

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, XGBoost, MLflow, Databricks, Delta Lake, and Databricks Model Serving.

## Limitations

- Only 24 experiments contain complete MTC and label data.
- The dataset is moderately imbalanced.
- Raw accelerometer data was not included in the baseline pipeline.
- Performance may vary across unseen machining experiments.

## Future Improvements

- add accelerometer time-domain and frequency-domain features
- collect more labeled industrial experiments
- apply cost-sensitive evaluation
- monitor prediction and feature drift
- automate endpoint updates after successful model promotion

## Evidence

Screenshots documenting the complete machine learning lifecycle are available in the `screenshots` directory.
