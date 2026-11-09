import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBClassifier, XGBRegressor

#1. data
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. model
model = XGBRegressor(random_state=123)

#3,4. train, evaluate

scores = cross_val_score(model, x, y, cv=kfold)
print('acc : ', scores, '\n 평균 acc : ', round(np.mean(scores),2))



















