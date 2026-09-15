import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense ,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
# import matplotlib.pyplot as plt
import time
import my_util
#1.data
datasets = load_diabetes()
x = datasets.data #딕셔너리 개념
y = datasets.target #target 이란 개념에 y데이터 모여있음

# print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=273
)

from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #train의 xmin,xmax학습 후 변환시킴 모두 0~1사이로
x_test = scaler.transform(x_test)

#2.model
# model = Sequential()
# model.add(Dense(120, input_dim=10))
# model.add(Dropout(0.4))
# model.add(Dense(30))
# model.add(Dense(48))
# model.add(Dropout(0.3))
# model.add(Dense(23))
# model.add(Dense(1))

#함수형모델
input1 = Input(shape=(10,))
dense1 = Dense(120)(input1)
drop1 = Dropout(0.4)(dense1)
dense2 = Dense(30)(drop1)
dense3 = Dense(48)(dense2)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(23)(drop2)
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)
#3.compile,train
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1,)


# ################# mcp 세이브 파일명 만들기 시작 #####################
# import datetime
# date = datetime.datetime.now() #현재 시간반환

# print(type(date)) #<class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M") #0914_1148
# print(date)
# print(type(date)) #<class 'str'>


# path = './_save/keras31/'
# filename = '-{epoch:04d}-{val_loss:.4f}.keras'
# filepath = "".join([path, "k31_", date, filename]) #문자없이 빈공간. + join 뒤에 붙힌다. + 
# #'./_save/keras30/' + "k30_" + #0914_1148 + '530-0.001.keras'

# ################# mcp 세이브 파일명 만들기 끝 #####################

# exit()
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=filepath, verbose=1,)
start_time = time.time()

batch_size=40
history = model.fit(x_train,y_train, epochs=500, batch_size = 40,
                    callbacks=[es,],
                    verbose=1, validation_split=0.15)

train_time = time.time() - start_time
print("학습시간: ", train_time)


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
results = model.predict(x_test)
# print("예측값 :", results[:10])
# print("실제값 :", y_test[:10])


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








