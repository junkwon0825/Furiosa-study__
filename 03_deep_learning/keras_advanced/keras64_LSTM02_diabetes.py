import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import time
import my_util


#1.data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=273
)

scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)   # (353, 10)
print(x_test.shape)    # (89, 10)

# LSTM은 3차원
# (samples, timesteps, features)
x_train = x_train.reshape(-1, 10, 1)
x_test = x_test.reshape(-1, 10, 1)

print(x_train.shape)   # (353, 10, 1)
print(x_test.shape)    # (89, 10, 1)


#2.model
model = Sequential()

model.add(LSTM(64, input_shape=(10, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()


#3.compile,train
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)

start_time = time.time()

batch_size = 30

history = model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.15,
    callbacks=[es]
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

results = model.predict(x_test)

my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=273,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    csv_file_path="./keras/model_history_log_v2.csv"
)