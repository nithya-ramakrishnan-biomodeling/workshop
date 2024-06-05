import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from os import scandir
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib as mpl



def load_combined_data():
    data_csv = pd.read_csv('/Users/nitrama/Downloads/combined_meth_tpm_unstranded_all_cases.csv')

    y = data_csv[['methylation']]
    X = data_csv[['tpm_unstranded']]

    X = X.to_numpy()
    y = y.to_numpy()
    return [X,y]


def run_model():
    mpl.rcParams['agg.path.chunksize'] = 10000

    [X,y] = load_combined_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

    #model = SVR(kernel='rbf')
    model = XGBRegressor(n_jobs=30, n_estimators=800)

    # train the model
    model.fit(X_train, y_train)
    # prediction on a test set
    y_pred = model.predict(X_test)
    # compute the r2 score
    r2 = r2_score(y_test, y_pred)
    print("r2 score is %0.2f (closer to 1 is good) " % r2)



def cross_validate_normalize_predict():
    [X,y] = load_combined_data()
    cv=10
    kf = KFold(n_splits=cv)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
        print('Cross validation ', fold)
        print(train_index)
        print(test_index)
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        X_test_original = X_test

        # # # data standardization - mean 0, sd = 1  - https://www.geeksforgeeks.org/how-to-scale-pandas-dataframe-columns/
        ## No need for standardization as the data are either categorical and all months related data on a same scale
        # same results with or without standardization
        std_scaler = StandardScaler()  # MinMaxScaler() #StandardScaler()
        X_train = std_scaler.fit_transform(X_train)  # X_train.to_numpy()
        X_test = std_scaler.transform(X_test)  # X_test.to_numpy()
        model = LinearRegression()

        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        print("r2 score is %0.2f (closer to 1 is good) " % r2)

if __name__ == "__main__":
    #cross_validate_normalize_predict()
    run_model()