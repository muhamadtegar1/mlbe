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


# Custom Transformers (needed for loading saved models)
class CustomImputer(BaseEstimator, TransformerMixin):
    """Transformer kustom untuk menerapkan logika imputation rule-based."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Logika Imputasi untuk 'unknown' (Job & Education)
        X_copy.loc[(X_copy['age']>60) & (X_copy['job']=='unknown'), 'job'] = 'retired'
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

        # Transformasi Bulan
        month_map = {'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        if 'month' in X_copy.columns:
            X_copy['month_num'] = X_copy['month'].map(month_map)
            X_copy['month_sin'] = np.sin(2 * np.pi * X_copy['month_num']/12)
            X_copy['month_cos'] = np.cos(2 * np.pi * X_copy['month_num']/12)
            X_copy.drop(columns=['month', 'month_num'], inplace=True)

        # Transformasi Hari
        day_map = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}
        if 'day_of_week' in X_copy.columns:
            X_copy['day_num'] = X_copy['day_of_week'].map(day_map)
            X_copy['day_sin'] = np.sin(2 * np.pi * X_copy['day_num']/5)
            X_copy['day_cos'] = np.cos(2 * np.pi * X_copy['day_num']/5)
            X_copy.drop(columns=['day_of_week', 'day_num'], inplace=True)

        return X_copy

    def get_feature_names_out(self, input_features=None):
        return ['month_sin', 'month_cos', 'day_sin', 'day_cos']


# CRITICAL: Register custom transformers in __main__ module for joblib unpickling
# This ensures joblib can find these classes when loading the pickled models
sys.modules['__main__'].CustomImputer = CustomImputer  # type: ignore
sys.modules['__main__'].CyclicalFeatureTransformer = CyclicalFeatureTransformer  # type: ignore

# Initialize FastAPI app
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API for predicting bank deposit subscriptions using Random Forest or CatBoost models",
    version="1.0.0"
)

# Load models at startup
try:
    rf_model = joblib.load("./random_forest_pipeline.joblib")
    print("✓ Random Forest model loaded successfully")
except Exception as e:
    print(f"✗ Error loading Random Forest model: {e}")
    rf_model = None

try:
    cb_classifier = CatBoostClassifier()
    cb_classifier.load_model("./catboost_model.cbm")
    # Load the preprocessing pipeline from Random Forest (same preprocessing for both)
    cb_preprocessor = joblib.load("random_forest_pipeline.joblib").named_steps['preprocessor_final']
    print("✓ CatBoost model loaded successfully")
except Exception as e:
    print(f"✗ Error loading CatBoost model: {e}")
    cb_classifier = None
    cb_preprocessor = None


# Pydantic models for request/response validation
class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age")
    job: str = Field(..., description="Job type (e.g., admin., blue-collar, entrepreneur, etc.)")
    marital: str = Field(..., description="Marital status (married, single, divorced)")
    education: str = Field(..., description="Education level (e.g., basic.4y, high.school, university.degree)")
    default: str = Field(..., description="Has credit in default? (yes, no, unknown)")
    housing: str = Field(..., description="Has housing loan? (yes, no, unknown)")
    loan: str = Field(..., description="Has personal loan? (yes, no, unknown)")
    contact: str = Field(..., description="Contact communication type (cellular, telephone)")
    month: str = Field(..., description="Last contact month (jan, feb, mar, etc.)")
    day_of_week: str = Field(..., description="Last contact day of week (mon, tue, wed, thu, fri)")
    duration: int = Field(..., ge=0, description="Last contact duration in seconds")
    campaign: int = Field(..., ge=1, description="Number of contacts during this campaign")
    pdays: int = Field(..., ge=0, description="Days since last contact (999 means never contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign")
    poutcome: str = Field(..., description="Outcome of previous campaign (failure, nonexistent, success)")
    emp_var_rate: float = Field(..., description="Employment variation rate")
    cons_price_idx: float = Field(..., description="Consumer price index")
    cons_conf_idx: float = Field(..., description="Consumer confidence index")
    euribor3m: float = Field(..., description="Euribor 3 month rate")
    nr_employed: float = Field(..., description="Number of employees")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 56,
                "job": "housemaid",
                "marital": "married",
                "education": "basic.4y",
                "default": "no",
                "housing": "no",
                "loan": "no",
                "contact": "telephone",
                "month": "may",
                "day_of_week": "mon",
                "duration": 261,
                "campaign": 1,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp_var_rate": 1.1,
                "cons_price_idx": 93.994,
                "cons_conf_idx": -36.4,
                "euribor3m": 4.857,
                "nr_employed": 5191.0
            }
        }
    )


class PredictionRequest(BaseModel):
    model: Literal["random_forest", "catboost"] = Field(..., description="Model to use for prediction")
    data: CustomerData | List[CustomerData] = Field(..., description="Single customer or list of customers for prediction")


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
    """Root endpoint with API information"""
    return {
        "message": "Bank Marketing Prediction API",
        "available_models": ["random_forest", "catboost"],
        "endpoints": {
            "/predict": "POST - Make predictions using selected model",
            "/health": "GET - Check API and model health",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
def health_check():
    """Check if models are loaded and API is healthy"""
    return {
        "status": "healthy",
        "models": {
            "random_forest": "loaded" if rf_model is not None else "not loaded",
            "catboost": "loaded" if cb_classifier is not None else "not loaded"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Make prediction using the selected model
    
    - **model**: Choose between 'random_forest' or 'catboost'
    - **data**: Single customer or list of customers for prediction
    
    Returns predictions (0/1), labels, and probability scores
    """
    
    # Convert to list if single customer
    customers = request.data if isinstance(request.data, list) else [request.data]
    
    # Convert customer data to DataFrame
    customers_list = []
    for customer in customers:
        customer_dict = customer.model_dump()
        # Rename keys to match expected column names
        customer_dict['emp.var.rate'] = customer_dict.pop('emp_var_rate')
        customer_dict['cons.price.idx'] = customer_dict.pop('cons_price_idx')
        customer_dict['cons.conf.idx'] = customer_dict.pop('cons_conf_idx')
        customer_dict['nr.employed'] = customer_dict.pop('nr_employed')
        customers_list.append(customer_dict)
    
    df = pd.DataFrame(customers_list)
    
    # Make predictions based on selected model
    if request.model == "random_forest":
        if rf_model is None:
            raise HTTPException(status_code=503, detail="Random Forest model not available")
        
        predictions = rf_model.predict(df)
        probabilities = rf_model.predict_proba(df)[:, 1]
        model_name = "Random Forest"
        
    elif request.model == "catboost":
        if cb_classifier is None or cb_preprocessor is None:
            raise HTTPException(status_code=503, detail="CatBoost model not available")
        
        # Preprocess data using the same pipeline
        df_processed = cb_preprocessor.transform(df)
        # Ensure predictable typing by explicitly creating a numpy array with dtype=int
        predictions = np.array(cb_classifier.predict(df_processed), dtype=int)  # type: ignore
        probabilities = cb_classifier.predict_proba(df_processed)[:, 1]  # type: ignore
        model_name = "CatBoost"
    
    else:
        raise HTTPException(status_code=400, detail="Invalid model selection")
    
    # Prepare response
    results = []
    for pred, prob in zip(predictions, probabilities):
        prediction_label = "Deposit (Yes)" if pred == 1 else "No Deposit (No)"
        results.append(SinglePrediction(
            prediction=int(pred),
            prediction_label=prediction_label,
            probability=round(float(prob), 4)
        ))
    
    return PredictionResponse(
        model_used=model_name,
        predictions=results,
        count=len(results),
        message=f"Prediction completed successfully using {model_name} for {len(results)} customer(s)"
    )


def main():
    """Run the FastAPI server"""
    print("\n" + "="*50)
    print("🚀 Starting Bank Marketing Prediction API")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
