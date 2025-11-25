"""
Custom transformers for bank marketing prediction models
These must be in a separate module for BentoML to load them correctly
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomImputer(BaseEstimator, TransformerMixin):
    """Transformer for rule-based imputation of missing values."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Rule-based imputation logic
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
    """Transform cyclical features (month, day_of_week) to sin and cos."""
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
