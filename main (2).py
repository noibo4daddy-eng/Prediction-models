from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# allow the HTML page to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


reg_metrics = joblib.load("regression_metrics.joblib")
clf_metrics = joblib.load("classification_metrics.joblib")

# load the saved files once, when the server starts
reg_model = joblib.load("regression_model.joblib")
reg_scaler = joblib.load("regression_scaler.joblib")
reg_columns = joblib.load("regression_columns.joblib")

clf_model = joblib.load("classification_model.joblib")
clf_scaler = joblib.load("classification_scaler.joblib")
clf_columns = joblib.load("classification_columns.joblib")


# the inputs each endpoint expects
class RegressionInput(BaseModel):
    year: int
    km_driven: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: float
    brand: str


class ClassificationInput(BaseModel):
    year: int
    selling_price: float
    km_driven: float
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: float
    brand: str


@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    # one row table from the input
    row = pd.DataFrame([data.dict()])
    # same preparation as training
    row = pd.get_dummies(row)
    row = row.reindex(columns=reg_columns, fill_value=0)
    row_scaled = reg_scaler.transform(row)
    # model predicts log price, so convert back to rupees
    price = float(np.exp(reg_model.predict(row_scaled)[0]))
    return {"predicted_price": round(price)}


@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    row = pd.DataFrame([data.dict()])
    row = pd.get_dummies(row)
    row = row.reindex(columns=clf_columns, fill_value=0)
    row_scaled = clf_scaler.transform(row)
    fuel = str(clf_model.predict(row_scaled)[0])
    return {"predicted_fuel": fuel}

@app.get("/stats")
def get_stats():
    return {"regression": reg_metrics, "classification": clf_metrics}