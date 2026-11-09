import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler


#1. data
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle=True,
    train_size=0.8,
    random_state=123
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=333
)


#2. model
model = DecisionTreeRegressor(
    random_state=123
)


#3,4. train, evaluate
scores = cross_val_score(
    model,
    x_train,
    y_train,
    cv=kfold
)

print('R2 : ', scores)
print('평균 R2 : ', round(np.mean(scores), 2))


# train 데이터에 대한 K-Fold 예측
y_predict_cv = cross_val_predict(
    model,
    x_train,
    y_train,
    cv=kfold
)

print('CV R2 : ', r2_score(y_train, y_predict_cv))


# 최종 test 평가
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print('test R2 : ', r2)