import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb

from configs import get_timezone

#setting timezone
get_timezone()

def get_sets(set):
    df = pd.read_csv(f'data/{set}.csv')
    y = pd.read_csv(f'data/y_{set}.csv')
    return df,y

def train_model_arima(train):
    model = sm.tsa.arima.ARIMA(train, order=(5,1,0))
    model = model.fit()
    #print(model.summary())
    return model

def train_model_xgb(train, y_train, test, y_test):
    params = {'max_depth':8, "booster": "gbtree", 'eta':0.1, 'objective':'reg:squarederror'}
    train = xgb.DMatrix(train, y_train)
    test = xgb.DMatrix(test, y_test)
    watchlist = [(train, 'train'), (test, 'eval')]
    # Training the model
    xgboost = xgb.train(params, train, 100, evals=watchlist, early_stopping_rounds= 20, verbose_eval=True)
    return xgboost

def evaluate_model_arima(model, test):
    forecast = model.forecast(len(test))
    mae = mean_absolute_error(test, forecast)
    mape = mean_absolute_percentage_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    # Add the evaluation results to a dictionary
    results = {'mae': mae, 'rmse': rmse, 'mape': mape}
    return results, forecast

def evaluate_model_xgb(model, test, y_test):
    test = xgb.DMatrix(test, y_test)
    preds = model.predict(test)
    rms_xgboost = np.sqrt(mean_squared_error(y_test, preds))
    mae_xgboost = mean_absolute_error(y_test, preds)
    mape_xgboost = mean_absolute_percentage_error(y_test, preds)
    # Add the evaluation results to a dictionary
    results = {'mae': mae_xgboost, 'rmse': rms_xgboost, 'mape': mape_xgboost}
    return results, preds

def main():

    print(f'Started Train step at {datetime.now().strftime("%H:%M:%S")}')

    # Read train and test sets
    print(f'Reading train and test sets')
    train, y_train = get_sets('train')
    test, y_test = get_sets('test')
    
    #Train model
    print(f'Training dataset')
    model = train_model_xgb(train, y_train, test, y_test)
    
    # Get metrics and forecasted values
    print(f'Getting metrics')
    metrics, predictions = evaluate_model_xgb(model, test, y_test)    
    print(metrics)
       
    # Save model
    print(f'Saving model')
    joblib.dump(model, 'model.pkl')
    
    features = list(model.get_score(importance_type='gain').keys())
    print(f"Features:{features}")

    print(f'Finished train step at {datetime.now().strftime("%H:%M:%S")}')

if __name__ == "__main__":

    main()