import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

#1. data
path = './data/kaggle_santander/'

train = pd.read_csv(path + 'train.csv')

x = train.drop(['TARGET', 'ID'], axis=1)
y = train['TARGET']

n_splits = 5
kfold = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=333
)

#2. model
model = XGBClassifier(
    random_state=123,
    eval_metric='logloss'
)

#3,4. train, evaluate
scores = cross_val_score(
    model,
    x,
    y,
    cv=kfold
)

print('acc : ', scores)
print('평균 acc : ', round(np.mean(scores), 4))