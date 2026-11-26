#1.data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import time
import my_util

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
# (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=49
)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)
# (16512, 8)

print(x_test.shape)
# (4128, 8)

# LSTM 입력은 3차원
x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)

print(x_train.shape)
# (16512, 8, 1)

print(x_test.shape)
# (4128, 8, 1)


#2.model
model = Sequential()

model.add(LSTM(64, input_shape=(8, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()


#3.compile,train
model.compile(
    loss='mse',
    optimizer='adam'
)

start_time = time.time()

batch_size = 40

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.15
)

train_time = time.time() - start_time


#4.evaluate,predict
loss = model.evaluate(x_test, y_test)

print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("MSE :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=49,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    csv_file_path="./keras/model_history_log_v2.csv"
)