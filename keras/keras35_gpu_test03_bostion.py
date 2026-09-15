from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Input
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
import time
import my_util
import numpy as np
#1.data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


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
# model.add(Dense(10, input_dim=13))
# model.add(Dropout(0.4))
# model.add(Dense(30))
# model.add(Dense(62))
# model.add(Dense(48))
# model.add(Dropout(0.4))
# model.add(Dense(23))
# model.add(Dropout(0.4))
# model.add(Dense(1))
##함수형 모델
input1 = Input(shape=(13,))
dense1 = Dense(10)(input1)
drop1 = Dropout(0.4)(dense1)
dense2 = Dense(30)(drop1)
dense3 = Dense(62)(dense2)
dense4 = Dense(48)(dense3)
drop2 = Dropout(0.4)(dense4)
dense5 = Dense(23)(drop2)
drop3 = Dropout(0.4)(dense5)
output1 = Dense(1)(drop3)

model = Model(inputs=input1, outputs=output1)
#3.compile,train
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1,)


################# mcp 세이브 파일명 만들기 시작 #####################
# import datetime
# date = datetime.datetime.now() #현재 시간반환

# print(type(date)) #<class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M") #0914_1148
# print(date)
# print(type(date)) #<class 'str'>


# path = './_save/keras32/'
# filename = '-{epoch:04d}-{val_loss:.4f}.keras'
# filepath = "".join([path, "k32_", date, filename]) #문자없이 빈공간. + join 뒤에 붙힌다. + 
# #'./_save/keras30/' + "k30_" + #0914_1148 + '530-0.001.keras'

################# mcp 세이브 파일명 만들기 끝 #####################

# exit()
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=filepath, verbose=1,)
start_time = time.time()

batch_size=40
history = model.fit(x_train,y_train, epochs=500, batch_size = 40,
                    callbacks=[es,],
                    verbose=1, validation_split=0.15)

train_time = time.time() - start_time
print(train_time)

print("====================================================")

#4.evaluate, predict
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


#4.042327165603638