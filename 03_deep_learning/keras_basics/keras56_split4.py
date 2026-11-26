import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GRU, LSTM


#1. data
a = np.array(range(1,101))

size = 10


def split_xy(dataset, size):
    x = []
    y = []

    for i in range(len(dataset) - size):
        x_subset = dataset[i:i+size]
        y_subset = dataset[i+size]

        x.append(x_subset)
        y.append(y_subset)

    return np.array(x), np.array(y)


x, y = split_xy(a, size)
x = x.reshape(x.shape[0], 5, 2)
print(x.shape)
# (90,5,2)
# print(x)
# print(y)

# exit()
#2. model
model = Sequential()

model.add(GRU(64, input_shape=(5,2)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


#3. compile, train
learning_rate = 0.001

model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=learning_rate)
)

model.fit(
    x,
    y,
    epochs=1000
)


#4. evaluate, predict
results = model.evaluate(x, y)
print('loss : ', results)

x_predict = np.array(range(91,106))
# 91~105에서 10개씩 자름
def split_x(dataset, size):
    aaa = []

    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i+size]
        aaa.append(subset)

    return np.array(aaa)
x_predict = split_x(x_predict, 10)

print(x_predict)
print(x_predict.shape)

x_predict = x_predict.reshape(x_predict.shape[0],5,2)

print(x_predict.shape)
# (6,5,2)
y_predict = model.predict(x_predict)

print('101~106 예측 결과 : ', y_predict)
