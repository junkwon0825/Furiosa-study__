#27-1 copy
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt #그래프 같은거 그릴때
# import matplotlib.font_manager as fm
import time
import my_util
from sklearn.datasets import fetch_california_housing, load_iris #data 제공됨 여러개
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

#통상적으로 import하는 애들은 위로 올림

#1.data
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target
print(x.shape, y.shape) 

"""
MinMaxScaler

(x - x_min) / (max - min) 

"""

x_train, x_test, y_train, y_test = train_test_split(#x_train,x_test이거 순서중요
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=49
)

from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) #train의 xmin,xmax학습 후 변환시킴 모두 0~1사이로
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train), np.max(x_train)) #0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test)) #-0.0010638297872338498 1.0
print(x_test.shape,x_train.shape,y_test.shape,y_train.shape) #(4128, 8) (16512, 8) (4128,) (16512,)
x_train = x_train.reshape(-1, 2, 4, 1)
x_test  = x_test.reshape(-1, 2, 4, 1)

#2.model
model = Sequential()
model.add(Conv2D(64,(1,2),input_shape=(2,4,1), activation='relu', padding='same')) #(26,26,64)
model.add(Conv2D(filters=32, kernel_size=(1,2), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(30,(1,2),activation='relu', padding='same')) #(20,20,16)
model.add(Conv2D(30,(1,2),activation='relu', padding='same')) #(20,20,16)
model.add(Conv2D(30,(1,2),activation='relu', padding='same')) #(21,21,16)

model.add(Flatten()) #2차원변경
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(1, activation='softmax')) # (10,)
model.summary()

#3.compile,train
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

batch_size=40
history = model.fit(x_train,y_train, epochs=100, batch_size = 30,
                    verbose=1, validation_split=0.15)

train_time = time.time() - start_time
print("====================================================")

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

# R2 기준 0.55 
# r2 :  0.5207760566600557
# r2 :  0.779879422449819 (성능향상 굳)



