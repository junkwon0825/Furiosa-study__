import pandas as pd
import numpy as np
import time
import my_util

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error


#1.data
path = "./_data/kaggle_bike/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.isna().sum())

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

print(x.shape)     # (10886, 8)
print(y.shape)     # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=78
)

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)   # (8708, 8)
print(x_test.shape)    # (2178, 8)

# LSTM용 3차원
x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)

print(x_train.shape)   # (8708, 8, 1)
print(x_test.shape)    # (2178, 8, 1)


#2.model
model = Sequential()

model.add(LSTM(64, input_shape=(8, 1)))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(1))


#3.compile,train
model.compile(
    loss='mse',
    optimizer='adam'
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

start_time = time.time()

batch_size = 55

history = model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.15,
    callbacks=[es]
)

train_time = time.time() - start_time


#4.evaluate,predict
loss = model.evaluate(x_test, y_test)

print("loss :", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print("r2 :", r2)
print("MSE :", mse)
print(f"RMSE : {rmse:.2f}")


# submission 데이터도 동일하게 전처리
test_csv = scaler.transform(test_csv)

test_csv = test_csv.reshape(-1, 8, 1)

print(test_csv.shape)

y_submit = model.predict(test_csv)

y_submit = y_submit.reshape(-1,)

submission['count'] = y_submit

submission.to_csv(
    path + "submit/" + "submit_LSTM.csv",
    index=True
)


my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=78,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    csv_file_path="./keras/model_history_log_v2.csv"
)