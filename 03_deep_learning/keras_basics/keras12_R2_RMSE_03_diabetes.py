#11-1copy
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time
import my_util

#1.data
datasets = load_diabetes()
x = datasets.data #딕셔너리 개념
y = datasets.target #target 이란 개념에 y데이터 모여있음

print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    random_state=48
)

#2.model
model = Sequential()
model.add(Dense(256, input_dim=10))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


#3.compile,train
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

batch_size=23
history = model.fit(x_train,y_train, epochs=340, batch_size = 23)

train_time = time.time() - start_time
print("================================================")
#4.evaluate,predict
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)
print("MSE : ", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2 = r2
)


#R2 기준 0.62이상






