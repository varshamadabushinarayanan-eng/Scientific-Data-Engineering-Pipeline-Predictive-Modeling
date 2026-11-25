# Project 1 — Scientific Data Engineering: Reproducible Data Pipeline & Predictive Modeling

## What this demonstrates
A reproducible Python workflow using a **synthetic IVD assay dataset** (650 rows) to:
- validate and clean tabular scientific observations;
- engineer features and encode categorical variables;
- train a regression model to estimate assay signal;
- train a classifier to predict pass/fail status;
- evaluate with R², MAE, MSE, RMSE and classification metrics;
- flag unusual observations with Isolation Forest;
- persist outputs and a SQLite copy of the dataset;
- expose predictions through FastAPI.

## Important data note
`data/synthetic_ivd_assay_data.csv` is generated demonstration data, not Biosensia/Axxelent/company data, not patient data, and not experimental evidence. The `synthetic_anomaly_flag` column is included only for checking the anomaly demonstration; do not use it as a model input.

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
uvicorn src.api:app --reload
```
Open `http://127.0.0.1:8000/docs` for the API.

Run from this folder. Training creates `artifacts/` with fitted models, metrics, predictions and `scientific_data.db`.

## Example API payload
```json
{
  "sample_type": "serum",
  "instrument_id": "reader_A",
  "reagent_lot": "LOT_101",
  "analyte_concentration": 4.2,
  "incubation_minutes": 12.0,
  "temperature_c": 25.0,
  "background_ratio": 0.12,
  "hemolysis_index": 0.2
}
```
