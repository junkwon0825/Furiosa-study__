import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler


#1. data
path = './data/kaggle_santander/'

datasets = pd.read_csv(path + 'train.csv')

x = datasets.drop(['TARGET', 'ID'], axis=1)
y = datasets['TARGET']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle=True,
    train_size=0.8,
    random_state=123,
    stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5

kfold = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=333
)


#2. model
model = DecisionTreeClassifier(
    random_state=123
)


#3,4. train, evaluate
scores = cross_val_score(
    model,
    x_train,
    y_train,
    cv=kfold
)

print('acc : ', scores)
print('평균 acc : ', round(np.mean(scores), 2))


y_predict_cv = cross_val_predict(
    model,
    x_train,
    y_train,
    cv=kfold
)

print('CV acc : ', accuracy_score(y_train, y_predict_cv))


# 최종 test
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print(y_predict)
print(y_test.values)

acc = accuracy_score(y_test, y_predict)

print('test acc : ', acc)