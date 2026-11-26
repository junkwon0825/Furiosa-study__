
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

#2.model
model = Sequential()
model.add(Dense(256, input_dim=9))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3.compile,train
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

batch_size = 15
history = model.fit(x_train,y_train, epochs=345, batch_size = 15,
                    verbose=1, validation_split=0.33)

train_time = time.time() - start_time

#4.evaluate, predict 평가,예측ㄱㄱ
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

test_csv = test_csv.fillna(test_csv.mean())
y_submit = model.predict(test_csv)
submission['count'] = y_submit

submission.to_csv(path + "submission_result.csv", index=True)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 79,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2 = r2
)