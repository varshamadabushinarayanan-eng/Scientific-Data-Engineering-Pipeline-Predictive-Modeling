from pathlib import Path
import json, sqlite3
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_ivd_assay_data.csv"
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
required = ["sample_id","sample_type","instrument_id","reagent_lot","analyte_concentration",
            "incubation_minutes","temperature_c","background_ratio","hemolysis_index",
            "assay_signal","release_status"]
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")
df = df.drop_duplicates(subset="sample_id").copy()
numeric = ["analyte_concentration","incubation_minutes","temperature_c","background_ratio","hemolysis_index"]
for col in numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["assay_signal","release_status"])

cat = ["sample_type","instrument_id","reagent_lot"]
features = numeric + cat
X = df[features]
y_reg = df["assay_signal"]
y_cls = df["release_status"].map({"fail":0,"pass":1})

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                      ("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat)
])
X_train, X_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg = Pipeline([("prep",preprocess),("model",RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2))])
reg.fit(X_train, yr_train)
reg_pred = reg.predict(X_test)
mse = mean_squared_error(yr_test, reg_pred)
reg_metrics = {"r2":float(r2_score(yr_test,reg_pred)), "mae":float(mean_absolute_error(yr_test,reg_pred)),
               "mse":float(mse), "rmse":float(np.sqrt(mse))}

X_train_c, X_test_c, yc_train, yc_test = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
clf = Pipeline([("prep",preprocess),("model",RandomForestClassifier(n_estimators=250, random_state=42,
                                                                   class_weight="balanced", min_samples_leaf=2))])
clf.fit(X_train_c, yc_train)
cls_pred = clf.predict(X_test_c)
cls_metrics = {"accuracy":float(accuracy_score(yc_test,cls_pred)), "f1":float(f1_score(yc_test,cls_pred))}
cls_report = classification_report(yc_test,cls_pred, target_names=["fail","pass"], output_dict=True)

# Unsupervised anomaly detection: excludes the demonstration label.
anomaly_model = Pipeline([("prep",preprocess),("model",IsolationForest(contamination=0.04, random_state=42))])
anomaly_model.fit(X)
df["model_anomaly_flag"] = (anomaly_model.predict(X) == -1).astype(int)

joblib.dump(reg, OUT/"regression_model.joblib")
joblib.dump(clf, OUT/"classification_model.joblib")
joblib.dump(anomaly_model, OUT/"anomaly_model.joblib")
(OUT/"metrics.json").write_text(json.dumps({"regression":reg_metrics,"classification":cls_metrics,
                                            "classification_report":cls_report}, indent=2))
df.to_csv(OUT/"observations_with_anomaly_flags.csv", index=False)
preds = pd.DataFrame({"actual_signal":yr_test.values,"predicted_signal":reg_pred})
preds.to_csv(OUT/"regression_test_predictions.csv",index=False)
with sqlite3.connect(OUT/"scientific_data.db") as conn:
    df.to_sql("assay_observations", conn, if_exists="replace", index=False)
print(json.dumps({"rows_used":len(df),"regression":reg_metrics,"classification":cls_metrics,
                  "artifacts_directory":str(OUT)},indent=2))
