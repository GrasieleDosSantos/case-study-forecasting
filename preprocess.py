import csv
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

#from configs import get_data_path, get_timezone, get_threshold, get_target
from configs import *
#setting timezone
get_timezone()

def get_dataset():
    dataset = pd.read_csv(get_path())
    return dataset

def split_data_arima(df, threshold):
    """ Uses a threshold to construct a mask when splitting a time ordered dataset """
    msk = (df.index < threshold)
    train = df[msk].copy()
    test = df[~msk].copy()
    return train, test

def split_data_xgb(df, target):
    """Usual split for feeding the xgboost algorithm"""
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def handle_missing_values(df):
    """ Drops a column if it has more than 40% of missing values and drops rows with null values """
    for col in df.columns:
        if df[col].isna().sum()/len(df) > 0.4:
            df.drop(col, axis=1, inplace=True)
    df.dropna(inplace=True)        
    return df

def set_and_sort_data_index(df, time_var):
    """ Sets the time variable as index and sort it """
    df.set_index(time_var, inplace=True)
    df.sort_index(inplace=True)
    return df

def get_series(train, test, y):
    """ Gets the time series values """
    train = train.loc[:, y]
    test =test.loc[:, y]
    return train, test

def main():
    
    print(f'Started preprocess step at {datetime.now().strftime("%H:%M:%S")}')
    df = get_dataset()
    
    print(f'Handling missing values')
    df = handle_missing_values(df)
    
    print("Encoding categorical variables")
    df = pd.get_dummies(data=df, columns=['DES_CATEGORIA_MATERIAL','DES_MARCA_MATERIAL'], drop_first=True)
    
    # Split dataset into training and test sets
    target = get_target()
    train, test, y_train, y_test = split_data_xgb(df, target)

    print(f'Train set with {len(train)} rows')
    print(f'Test set with {len(test)} rows')
    
    print("Dropping unecessary columns")
    cols_to_drop = columns_to_drop()
    train = train.drop(cols_to_drop, axis=1)
    test = test.drop(cols_to_drop, axis=1)
    
    # Save sets to csv
    print(f'Saving sets to csv')
    train.to_csv(f'data/train.csv', index=False, header=True, encoding='utf-8')
    test.to_csv(f'data/test.csv', index=False, header=True, encoding='utf-8')
    y_train.to_csv(f'data/y_train.csv', index=False, header=True, encoding='utf-8')
    y_test.to_csv(f'data/y_test.csv', index=False, header=True, encoding='utf-8')

    print(f'Finished Preprocess step at {datetime.now().strftime("%H:%M:%S")}')

if __name__ == "__main__":

    main()