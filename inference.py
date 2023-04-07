# custom modules
from configs import get_timezone, columns_to_drop
# external modules
import pickle
import pandas as pd
from datetime import datetime
import xgboost as xgb
import numpy as np

get_timezone()

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
# Get features names
default_features = model.feature_names

def prep_data(df, features):
    # Dropping unecessary columns
    df.drop(columns={'PCT_DESCONTO', 'COD_MATERIAL', 'QT_VENDA'}, inplace=True)
    # Dropping missing values
    df.dropna(inplace=True)
    # Encoding categorical data
    df_enc = pd.get_dummies(data=df, columns=['DES_CATEGORIA_MATERIAL','DES_MARCA_MATERIAL'], drop_first=True)
    
    # Create a set of expected columns based on the features list
    expected_cols = set(features)
    # Create a set of the actual columns in df_enc
    actual_cols = set(df_enc.columns)

    # Find any missing columns and add them with default values of zero
    missing_cols = expected_cols - actual_cols
    for col in missing_cols:
        df_enc[col] = 0

    # Drop any extra columns in df_enc that are not in the features list
    extra_cols = actual_cols - expected_cols
    df_enc.drop(columns=extra_cols, inplace=True)

    # Reorder columns in df_enc to match the order in the features list
    df_enc = df_enc[features]
    
    return df_enc
    
def predict_arima(new_cycles, len_cycle):
    """Receives a list of integers that correspond to new cycles and save the forecasted values to a xlsx file"""
    predictions = []
    for cycle in new_cycles:
        forecasted = model.forecast(len_cycle)
        predictions.append(forecasted)
    df = pd.DataFrame(predictions, columns=['Forecasted Values'])
    df.to_excel('data/forecasted_values.xlsx', index=False)
    return predictions

def predict_xgb(new_data, features = default_features, model = model):
    df = prep_data(new_data, features)
    df_xgb = xgb.DMatrix(df)
    predictions = model.predict(df_xgb)
    preds = pd.DataFrame(predictions, columns=['Predicted Values'])
    preds.to_excel('data/forecasted_values.xlsx', index=False)
    return preds

