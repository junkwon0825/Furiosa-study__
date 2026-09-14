#30-1 copy
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt #그래프 같은거 그릴때
# import matplotlib.font_manager as fm
import time
import my_util
from sklearn.datasets import fetch_california_housing, load_iris #data 제공됨 여러개
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

path = './_save/keras30/'

#1.data
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target
print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(#x_train,x_test이거 순서중요
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=49
)

from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) #train의 xmin,xmax학습 후 변환시킴 모두 0~1사이로
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train), np.max(x_train)) #0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test)) #-0.0010638297872338498 1.0


# 2.model
# model = Sequential()
# model.add(Dense(256, input_dim=8))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(4))
# model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model = load_model(path + 'keras30_mcp1.keras')

#3.compile,train
# model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1,)

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=path+ 'keras30_mcp1.keras', verbose=1,)
# start_time = time.time()

# batch_size=40
# history = model.fit(x_train,y_train, epochs=500, batch_size = 40,
#                     callbacks=[es,mcp],
#                     verbose=1, validation_split=0.15)

# train_time = time.time() - start_time


#4.evaluate,predict
loss = model.evaluate(x_test,y_test) #batch_size=32
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
    model=model,
    data_shape=x_train.shape,
    random_num=78,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    # test_loss=result,
    csv_file_path="./keras/model_history_log_v2.csv"
)


# r2 :  0.7844464385460788
# MSE :  0.27892186136062047
# RMSE :  0.5281305343952577