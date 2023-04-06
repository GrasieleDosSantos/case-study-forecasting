# custom modules
from configs import get_timezone, columns_to_drop
# external modules
import joblib
import pandas as pd
from datetime import datetime
import xgboost as xgb
import sys
import json

# Get features names
#features = list(model.get_score(importance_type='gain').keys())
#print(features)

def prep_data(df, features):
    # Dropping unecessary columns
    df.drop(columns={'PCT_DESCONTO', 'COD_MATERIAL', 'QT_VENDA'}, inplace=True)
    # Dropping missing values
    df.dropna(inplace=True)
    # Encoding categorical data
    df_enc = pd.get_dummies(data=df, columns=['DES_CATEGORIA_MATERIAL','DES_MARCA_MATERIAL'], drop_first=True)
    for i in features:
        if i not in df_enc.columns:
            print(f'Adding empty feature: {i}')
            df_enc[i] = False
            
    for i in df_enc.columns:
        if i not in features:
            print(f'Removing extra feature not in feature list: {i}')
            df_enc.drop(columns={i}, axis=1, inplace=True)
    
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

def predict_xgb(new_data, features):
    df = prep_data(new_data, features)
    df = xgb.DMatrix(df)
    model = joblib.load('model.pkl')
    predictions = model.predict(df)
    preds = pd.DataFrame(predictions, columns=['Predicted Values'])
    preds.to_excel('data/forecasted_values.xlsx', index=False)
    return preds

