from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"artifacts"
app = FastAPI(title="Scientific Assay Prediction API", version="1.0")

class AssayInput(BaseModel):
    sample_type: str
    instrument_id: str
    reagent_lot: str
    analyte_concentration: float = Field(ge=0)
    incubation_minutes: float = Field(gt=0)
    temperature_c: float
    background_ratio: float = Field(ge=0)
    hemolysis_index: float = Field(ge=0, le=1)

def load_model(name):
    path = ART/name
    if not path.exists():
        raise HTTPException(status_code=503, detail="Train models first: run python src/train.py")
    return joblib.load(path)

@app.get("/")
def home():
    return {"message":"Synthetic IVD assay modelling API","docs":"/docs"}

@app.post("/predict")
def predict(payload: AssayInput):
    row = pd.DataFrame([payload.model_dump()])
    reg = load_model("regression_model.joblib")
    clf = load_model("classification_model.joblib")
    signal = float(reg.predict(row)[0])
    status = "pass" if int(clf.predict(row)[0]) == 1 else "fail"
    return {"predicted_assay_signal":round(signal,4),"predicted_release_status":status,
            "note":"Demonstration prediction only; not for clinical or product-release decisions."}
