"""
BentoML Service for Bank Marketing Prediction
Supports both Random Forest and CatBoost models
"""
from __future__ import annotations
import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List, Dict, Any

# Import custom transformers from separate module
# This ensures they're in sys.modules when joblib unpickles them
import transformers  # noqa: F401


# Pydantic Models
class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age")
    job: str = Field(..., description="Job type")
    marital: str = Field(..., description="Marital status")
    education: str = Field(..., description="Education level")
    default: str = Field(..., description="Has credit in default?")
    housing: str = Field(..., description="Has housing loan?")
    loan: str = Field(..., description="Has personal loan?")
    contact: str = Field(..., description="Contact communication type")
    month: str = Field(..., description="Last contact month")
    day_of_week: str = Field(..., description="Last contact day of week")
    duration: int = Field(..., ge=0, description="Last contact duration in seconds")
    campaign: int = Field(..., ge=1, description="Number of contacts during this campaign")
    pdays: int = Field(..., ge=0, description="Days since last contact")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign")
    poutcome: str = Field(..., description="Outcome of previous campaign")
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


class SinglePrediction(BaseModel):
    prediction: int
    prediction_label: str
    probability: float


class PredictionResponse(BaseModel):
    model_used: str
    predictions: List[SinglePrediction]
    count: int
    message: str

class PredictInput(BaseModel):
    model: Literal["random_forest", "catboost"]
    data: List[CustomerData]
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model": "random_forest",
            "data": [
                {
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
            ]
        }
    })


@bentoml.service(
    resources={
        "cpu": "2",
        "memory": "2Gi"
    },
    traffic={
        "timeout": 300,
        "concurrency": 32
    }
)
class BankMarketingPredictor:
    """BentoML Service for Bank Marketing Prediction with Random Forest and CatBoost models"""

    # Reference models from BentoML Model Store
    rf_model_ref = bentoml.models.BentoModel("bank_random_forest:latest")
    cb_model_ref = bentoml.models.BentoModel("bank_catboost:latest")

    def __init__(self):
        
        # Load models using BentoML's framework integrations
        # This properly handles custom transformers
        self.rf_model = bentoml.sklearn.load_model(self.rf_model_ref)
        self.cb_classifier = bentoml.catboost.load_model(self.cb_model_ref)
        # Extract preprocessor from the loaded RF pipeline
        self.cb_preprocessor = self.rf_model.named_steps['preprocessor_final'] # type: ignore

    def _prepare_dataframe(self, customers: List[CustomerData]) -> pd.DataFrame:
        """Convert customer data to DataFrame with correct column names"""
        customers_list = []
        for customer in customers:
            customer_dict = customer.model_dump()
            # Rename keys to match expected column names (with dots)
            customer_dict['emp.var.rate'] = customer_dict.pop('emp_var_rate')
            customer_dict['cons.price.idx'] = customer_dict.pop('cons_price_idx')
            customer_dict['cons.conf.idx'] = customer_dict.pop('cons_conf_idx')
            customer_dict['nr.employed'] = customer_dict.pop('nr_employed')
            customers_list.append(customer_dict)
        return pd.DataFrame(customers_list)

    @bentoml.api
    def predict_random_forest(self, data: CustomerData | List[CustomerData]) -> PredictionResponse:
        """
        Make predictions using Random Forest model
        
        Args:
            data: Single customer or list of customers
            
        Returns:
            PredictionResponse with predictions, probabilities, and metadata
        """
        customers = data if isinstance(data, list) else [data]
        df = self._prepare_dataframe(customers)
        
        predictions = self.rf_model.predict(df) # type: ignore
        probabilities = self.rf_model.predict_proba(df)[:, 1] # type: ignore
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            prediction_label = "Deposit (Yes)" if pred == 1 else "No Deposit (No)"
            results.append(SinglePrediction(
                prediction=int(pred),
                prediction_label=prediction_label,
                probability=round(float(prob), 4)
            ))
        
        return PredictionResponse(
            model_used="Random Forest",
            predictions=results,
            count=len(results),
            message=f"Prediction completed successfully using Random Forest for {len(results)} customer(s)"
        )

    @bentoml.api
    def predict_catboost(self, data: CustomerData | List[CustomerData]) -> PredictionResponse:
        """
        Make predictions using CatBoost model
        
        Args:
            data: Single customer or list of customers
            
        Returns:
            PredictionResponse with predictions, probabilities, and metadata
        """
        customers = data if isinstance(data, list) else [data]
        df = self._prepare_dataframe(customers)
        
        # Preprocess data
        df_processed = self.cb_preprocessor.transform(df)
        
        predictions = np.array(self.cb_classifier.predict(df_processed), dtype=int) 
        probabilities = self.cb_classifier.predict_proba(df_processed)[:, 1] # type: ignore
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            prediction_label = "Deposit (Yes)" if pred == 1 else "No Deposit (No)"
            results.append(SinglePrediction(
                prediction=int(pred),
                prediction_label=prediction_label,
                probability=round(float(prob), 4)
            ))
        
        return PredictionResponse(
            model_used="CatBoost",
            predictions=results,
            count=len(results),
            message=f"Prediction completed successfully using CatBoost for {len(results)} customer(s)"
        )

    @bentoml.api
    def predict(self, payload: PredictInput) -> PredictionResponse:
        """Unified prediction endpoint accepting a single structured payload.

        Args:
            payload: PredictInput containing model name and list of customer records.
        """
        print(f"[PREDICT] Starting prediction with model={payload.model}, records={len(payload.data)}")
        df = self._prepare_dataframe(payload.data)
        print(f"[PREDICT] DataFrame prepared with shape {df.shape}")

        if payload.model == "random_forest":
            print("[PREDICT] Using Random Forest model")
            predictions = self.rf_model.predict(df)  # type: ignore
            probabilities = self.rf_model.predict_proba(df)[:, 1]  # type: ignore
            model_name = "Random Forest"
        else:  # catboost
            print("[PREDICT] Using CatBoost model, preprocessing...")
            df_processed = self.cb_preprocessor.transform(df)
            print("[PREDICT] Preprocessing done, making predictions...")
            predictions = np.array(self.cb_classifier.predict(df_processed), dtype=int)
            probabilities = self.cb_classifier.predict_proba(df_processed)[:, 1]  # type: ignore
            model_name = "CatBoost"
        print(f"[PREDICT] Predictions complete, formatting results...")

        results = [
            SinglePrediction(
                prediction=int(pred),
                prediction_label="Deposit (Yes)" if pred == 1 else "No Deposit (No)",
                probability=round(float(prob), 4),
            )
            for pred, prob in zip(predictions, probabilities)
        ]

        return PredictionResponse(
            model_used=model_name,
            predictions=results,
            count=len(results),
            message=f"Prediction completed successfully using {model_name} for {len(results)} customer(s)",
        )

    @bentoml.api
    def health(self) -> Dict[str, str]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "random_forest": "loaded",
            "catboost": "loaded"
        }
