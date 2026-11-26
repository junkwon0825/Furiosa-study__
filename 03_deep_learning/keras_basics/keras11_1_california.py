import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing, load_iris #data 제공됨 여러개
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
#통상적으로 import하는 애들은 위로 올림

#1.data
datasets = fetch_california_housing()


x = datasets.data
y = datasets.target
print(x.shape, y.shape) #항상 이렇게 데이터 받으면 데이터 어떻게 생겼는지 찍어봄

x_train, x_test, y_train, y_test = train_test_split(#x_train,x_test이거 순서중요
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=49
)

#2.model
model = Sequential()
model.add(Dense(1, input_dim=8))
model.add(Dense(40))
model.add(Dense(89))
model.add(Dense(62))
model.add(Dense(48))
model.add(Dense(79))
model.add(Dense(1))

#3.compile,train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=385, batch_size=30)

print("====================================================")

#4.evaluate,predict
loss = model.evaluate(x_test,y_test) #batch_size=32
print('loss : ', loss)


results = model.predict(x_test)

print("예측값 :", results[:10].flatten())
print("실제값 :", y_test[:10])