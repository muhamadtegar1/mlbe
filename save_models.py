"""
Script to save trained models to BentoML Model Store
Run this script to register your models with BentoML before serving
"""
import bentoml
import sys
import os

# Add current directory to Python path so transformers module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import transformers BEFORE loading the pickled models
# Also import the classes directly into this namespace so joblib can find them
import transformers  # noqa: F401
from transformers import CustomImputer, CyclicalFeatureTransformer  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def save_models_to_bentoml():
    """Save Random Forest and CatBoost models to BentoML Model Store"""
    
    print("Loading Random Forest pipeline...")
    rf_pipeline = joblib.load("random_forest_pipeline.joblib")
    
    # Save Random Forest model using BentoML sklearn integration
    print("Saving Random Forest model to BentoML Model Store...")
    rf_model_tag = bentoml.sklearn.save_model(
        "bank_random_forest",
        rf_pipeline,
        labels={
            "model_type": "random_forest",
            "framework": "scikit-learn",
            "task": "classification"
        },
        metadata={
            "description": "Random Forest model for bank marketing prediction",
            "dataset": "bank-additional-full.csv",
            "preprocessing": "Custom imputation + cyclical features + standard scaling"
        }
    )
    print(f"✓ Random Forest model saved: {rf_model_tag}")
    
    # Save CatBoost model using bentoml.catboost integration
    print("\nLoading CatBoost model...")
    cb_classifier = CatBoostClassifier()
    cb_classifier.load_model("catboost_model.cbm")
    
    print("Saving CatBoost model to BentoML Model Store...")
    cb_model_tag = bentoml.catboost.save_model(
        "bank_catboost",
        cb_classifier,
        labels={
            "model_type": "catboost",
            "framework": "catboost",
            "task": "classification"
        },
        metadata={
            "description": "CatBoost model for bank marketing prediction",
            "dataset": "bank-additional-full.csv",
            "preprocessing": "Custom imputation + cyclical features + standard scaling"
        }
    )
    print(f"✓ CatBoost model saved: {cb_model_tag}")
    print("\n" + "="*60)
    print("✓ All models saved successfully to BentoML Model Store!")
    print("="*60)
    print("\nTo view saved models, run:")
    print("  bentoml models list")
    print("\nTo serve the models, run:")
    print("  bentoml serve service:BankMarketingPredictor")
    print("\nTo deploy to BentoCloud, run:")
    print("  bentoml deploy")


if __name__ == "__main__":
    save_models_to_bentoml()
