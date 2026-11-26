import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import time
import my_util


#1.data
path = "./_data/ddarung/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "submission.csv", index_col=0)

train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=79
)

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_test.shape, x_train.shape, y_test.shape, y_train.shape)
# (266, 9) (1062, 9) (266,) (1062,)

# LSTM은 3차원
x_train = x_train.reshape(-1, 9, 1)
x_test = x_test.reshape(-1, 9, 1)

print(x_train.shape)   # (1062, 9, 1)
print(x_test.shape)    # (266, 9, 1)


#2.model
model = Sequential()

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(9,1)))
model.add(Conv1D(10,2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(1))


#3.compile,train
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

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


# 제출 데이터
test_csv = test_csv.fillna(test_csv.mean())

test_csv = scaler.transform(test_csv)

# 이것도 LSTM shape으로 변경
test_csv = test_csv.reshape(-1, 9, 1)

print(test_csv.shape)

y_submit = model.predict(test_csv)

y_submit = y_submit.reshape(-1,)

submission['count'] = y_submit

submission.to_csv(path + "submission_result.csv", index=True)


my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=79,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    csv_file_path="./keras/model_history_log_v2.csv"
)