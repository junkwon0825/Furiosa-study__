import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler


#1. data
x, y = load_iris(return_X_y=True)
# print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size=0.8, random_state=123, stratify=y #회기에서 쓸 수 없음 
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. model
model = DecisionTreeClassifier()

#3,4. train, evaluate
scores = cross_val_score(model, x_train, y_train, cv=kfold)

print('acc : ', scores, '\n 평균 acc : ', round(np.mean(scores),2))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print("acc : ", acc)
