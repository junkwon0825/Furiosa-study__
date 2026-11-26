import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

#1. data
path = './data/kaggle_bike/'

train = pd.read_csv(path + 'train.csv')

# datetime은 문자열이라 제거
x = train.drop(['count', 'datetime'], axis=1)
y = train['count']

n_splits = 5
kfold = KFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=333
)

#2. model
model = XGBRegressor(
    random_state=123
)

#3,4. train, evaluate
scores = cross_val_score(
    model,
    x,
    y,
    cv=kfold
)

print('scores : ', scores)
print('평균 score : ', round(np.mean(scores), 4))