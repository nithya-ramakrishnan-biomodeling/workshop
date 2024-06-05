import os

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
import xgboost as xg
from os import scandir
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib as mpl
import shap


def load_csv():
    mod_csv = pd.read_csv('../../yeast_PTM.csv')
    print(mod_csv.columns.values)
    df_ML_X= mod_csv[['H2AK5ac','H2AS129ph','H3K14ac','H3K18ac']]
    df_ML_Y=mod_csv[['H3K23ac']]

    print(df_ML_Y)
    return [df_ML_X, df_ML_Y]


def shap_explain2(model, X_test, features):

    explainer = shap.TreeExplainer(model)
    # Calculates the SHAP values - It takes some time
    exp = explainer(X_test)
    shap_values = exp.values
    exp.feature_names = list(features)


    shap.plots.beeswarm(exp)

def main():

    [df_ML_X,df_ML_Y]=load_csv()
    X_train, X_test, y_train, y_test = train_test_split(df_ML_X, df_ML_Y, test_size = 0.30, random_state = 999)

    # train a model
    print('---- TRAINING -----')
    print("N = %d " % (len(df_ML_X)))

    # training a Linear Regression
    #model = LinearRegression()

    # # # training a SVM regressor
    #model = SVR(kernel='rbf')

    # # # training a XGBoost regressor
    model = XGBRegressor(n_jobs=8, n_estimators=100)

    # train the model
    model.fit(X_train, y_train.values.ravel())
    # prediction on a test set
    y_pred = model.predict(X_test)

    # compute the r2 score
    r2 = r2_score(y_test, y_pred)
    print("r2 score is %0.2f (closer to 1 is good) " % r2)

    features = ['H3K14ac', 'H3K14ac.1', 'H3K14ac.2', 'H3K14ac.3']
    # Explain using SHAP
    # shap_explain(model, transforms, X_test, X_test_original, y_test, y_pred_test, features)
    #shap_explain2(model, X_test.iloc[0:2],  features)
    shap_explain2(model, X_test, features)


def cross_validate_normalize_predict():
    mpl.rcParams['agg.path.chunksize'] = 10000

    [X,y] = load_csv()
    X = X.to_numpy()
    y = y.to_numpy()

    # train a model

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
        std_scalar = StandardScaler()
        X_train = std_scaler.fit_transform(X_train)
        X_test = std_scaler.transform(X_test)
        model = XGBRegressor(n_jobs=8, n_estimators=100)

        model.fit(X_train, y_train)
       # y_pred_train = model.predict(X_train)  # just for comparison purposes
        y_pred_test = model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        print("r2 score is %0.2f (closer to 1 is good) " % r2)




if __name__ == "__main__":
        main()
