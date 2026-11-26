import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#1.data
datasets = load_diabetes()
x = datasets.data #딕셔너리 개념
y = datasets.target #target 이란 개념에 y데이터 모여있음

print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.5,
    shuffle=True,
    random_state=273
)

#2.model
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(30))
model.add(Dense(70))
model.add(Dense(62))
model.add(Dense(48))
model.add(Dense(23))
model.add(Dense(1))


#3.compile,train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=345, batch_size = 30,
                    verbose=1, validation_split=0.33)

print("================================================")
#4.evaluate,predict
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

results = model.predict(x_test)
# print("예측값 :", results[:10])
# print("실제값 :", y_test[:10])












