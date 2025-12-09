from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List
import joblib  # type: ignore
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin
import uvicorn
import sys
import os

# Custom Transformers (needed for loading saved models)
class CustomImputer(BaseEstimator, TransformerMixin):
    """Transformer kustom untuk menerapkan logika imputation rule-based."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Logika Imputasi
        if 'age' in X_copy.columns and 'job' in X_copy.columns:
            X_copy.loc[(X_copy['age']>60) & (X_copy['job']=='unknown'), 'job'] = 'retired'
        
        if 'education' in X_copy.columns and 'job' in X_copy.columns:
            X_copy.loc[(X_copy['education']=='unknown') & (X_copy['job']=='management'), 'education'] = 'university.degree'
            X_copy.loc[(X_copy['education']=='unknown') & (X_copy['job']=='services'), 'education'] = 'high.school'
            X_copy.loc[(X_copy['education']=='unknown') & (X_copy['job']=='housemaid'), 'education'] = 'basic.4y'
            X_copy.loc[(X_copy['job'] == 'unknown') & (X_copy['education']=='basic.4y'), 'job'] = 'blue-collar'
            X_copy.loc[(X_copy['job'] == 'unknown') & (X_copy['education']=='basic.6y'), 'job'] = 'blue-collar'
            X_copy.loc[(X_copy['job'] == 'unknown') & (X_copy['education']=='basic.9y'), 'job'] = 'blue-collar'
            X_copy.loc[(X_copy['job']=='unknown') & (X_copy['education']=='professional.course'), 'job'] = 'technician'

        return X_copy


class CyclicalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Mengubah fitur siklus (month, day_of_week) menjadi sin dan cos."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        month_map = {'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        if 'month' in X_copy.columns:
            X_copy['month_num'] = X_copy['month'].map(month_map)
            X_copy['month_sin'] = np.sin(2 * np.pi * X_copy['month_num']/12)
            X_copy['month_cos'] = np.cos(2 * np.pi * X_copy['month_num']/12)
            X_copy.drop(columns=['month', 'month_num'], inplace=True)
            
        day_map = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}
        if 'day_of_week' in X_copy.columns:
            X_copy['day_num'] = X_copy['day_of_week'].map(day_map)
            X_copy['day_sin'] = np.sin(2 * np.pi * X_copy['day_num']/5)
            X_copy['day_cos'] = np.cos(2 * np.pi * X_copy['day_num']/5)
            X_copy.drop(columns=['day_of_week', 'day_num'], inplace=True)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return ['month_sin', 'month_cos', 'day_sin', 'day_cos']

# CRITICAL: Register custom transformers
sys.modules['__main__'].CustomImputer = CustomImputer 
sys.modules['__main__'].CyclicalFeatureTransformer = CyclicalFeatureTransformer 

app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API for predicting bank deposit subscriptions (Scope Down Version)",
    version="1.1.0"
)

# --- MODEL LOADING SECTION ---
rf_model = None
cb_classifier = None
cb_preprocessor = None

# Nama file model 
RF_MODEL_PATH = "models/model_lead_scoring_final_deployment.joblib"
CB_MODEL_PATH = "models/catboost_model.cbm" 

# 1. Load Random Forest (Pipeline Lengkap)
try:
    if os.path.exists(RF_MODEL_PATH):
        rf_model = joblib.load(RF_MODEL_PATH)
        print(f"✓ Random Forest model loaded from {RF_MODEL_PATH}")
    else:
        print(f"✗ File {RF_MODEL_PATH} not found!")
except Exception as e:
    print(f"✗ Error loading Random Forest model: {e}")

# 2. Load CatBoost & Ambil Preprocessor dari RF
try:
    if os.path.exists(CB_MODEL_PATH):
        cb_classifier = CatBoostClassifier()
        cb_classifier.load_model(CB_MODEL_PATH)
        print(f"✓ CatBoost model loaded from {CB_MODEL_PATH}")
        
        # PERBAIKAN: Ambil preprocessor dari rf_model yang baru, bukan load file lama
        if rf_model is not None:
            # Mengambil step 'preprocessor_final' dari pipeline RF
            cb_preprocessor = rf_model.named_steps['preprocessor_final']
            print("✓ Preprocessor extracted successfully from pipeline")
        else:
            print("✗ Cannot load preprocessor: RF Model is missing")
    else:
         print(f"✗ File {CB_MODEL_PATH} not found!")
except Exception as e:
    print(f"✗ Error loading CatBoost model: {e}")


# --- DATA MODELS ---
class CustomerData(BaseModel):
    # 10 Fitur Final Sesuai Scope Down
    age: int = Field(..., ge=18, le=100)
    job: str = Field(...)
    education: str = Field(...)
    month: str = Field(...)
    campaign: int = Field(..., ge=1)
    previous: int = Field(..., ge=0)
    poutcome: str = Field(...)
    cons_conf_idx: float = Field(...)
    euribor3m: float = Field(...)
    nr_employed: float = Field(...)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 30,
                "job": "admin.",
                "education": "university.degree",
                "month": "may",
                "campaign": 1,
                "previous": 0,
                "poutcome": "nonexistent",
                "cons_conf_idx": -36.4,
                "euribor3m": 4.857,
                "nr_employed": 5191.0
            }
        }
    )

class PredictionRequest(BaseModel):
    model: Literal["random_forest", "catboost"] = "random_forest"
    data: CustomerData | List[CustomerData]

class SinglePrediction(BaseModel):
    prediction: int
    prediction_label: str
    probability: float

class PredictionResponse(BaseModel):
    model_used: str
    predictions: List[SinglePrediction]
    count: int
    message: str

@app.get("/")
def read_root():
    return {"status": "active", "models_loaded": {
        "random_forest": rf_model is not None,
        "catboost": cb_classifier is not None
    }}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    customers = request.data if isinstance(request.data, list) else [request.data]
    
    customers_list = []
    for customer in customers:
        customer_dict = customer.model_dump()
        # Rename keys: Pydantic (_) to DataFrame (.)
        customer_dict['cons.conf.idx'] = customer_dict.pop('cons_conf_idx')
        customer_dict['nr.employed'] = customer_dict.pop('nr_employed')
        customers_list.append(customer_dict)
    
    df = pd.DataFrame(customers_list)
    
    # SAFETY: Pastikan urutan kolom sesuai training
    final_features = [
        'euribor3m', 'nr.employed', 'age', 'cons.conf.idx', 'campaign',
        'poutcome', 'previous', 'job', 'education', 'month'
    ]
    # Reorder columns (ignore if day_of_week is missing because we don't need it)
    df = df[final_features]

    if request.model == "random_forest":
        if rf_model is None:
            raise HTTPException(status_code=503, detail="Random Forest model not initialized")
        
        predictions = rf_model.predict(df)
        probabilities = rf_model.predict_proba(df)[:, 1]
        model_name = "Random Forest"
        
    elif request.model == "catboost":
        if cb_classifier is None or cb_preprocessor is None:
            raise HTTPException(status_code=503, detail="CatBoost model/preprocessor not initialized")
        
        df_processed = cb_preprocessor.transform(df)
        predictions = np.array(cb_classifier.predict(df_processed), dtype=int)
        probabilities = cb_classifier.predict_proba(df_processed)[:, 1]
        model_name = "CatBoost"
    else:
        raise HTTPException(status_code=400, detail="Invalid model")
    
    results = []
    for pred, prob in zip(predictions, probabilities):
        label = "Deposit (Yes)" if pred == 1 else "No Deposit (No)"
        results.append(SinglePrediction(
            prediction=int(pred),
            prediction_label=label,
            probability=round(float(prob), 4)
        ))
    
    return PredictionResponse(
        model_used=model_name,
        predictions=results,
        count=len(results),
        message="Success"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)