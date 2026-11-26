#28-1 copy
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
model = Sequential()
model.add(Dense(256, input_dim=8))
model.add(Dense(128,activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8,activation='relu'))
model.add(Dense(4))
model.add(Dense(1))

# model.summary()

path = './_save/keras29/'
# model.save(path + 'keras29_1_save_model.keras')
# model = load_model(path + 'keras29_1_save_model.keras')

model.summary()

#3.compile,train
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

batch_size=40
history = model.fit(x_train,y_train, epochs=100, batch_size = 30,
                    verbose=1, validation_split=0.15)

train_time = time.time() - start_time

model.save(path + 'keras29_3_save_model.keras')

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

# print("====================== history ===============================")
# print(history.history['loss'])
# print(history.history['val_loss'])
# plt.figure(figsize=(9,6)) #그냥 그림판 자체 크기 사이즈
# plt.plot(history.history['loss'][3:], c='red', label='loss') #y값만 넣으면 x디폴트는 시간순으로 그려줌
# plt.plot(history.history['val_loss'][3:], c='blue', label='val_loss')
# plt.legend(loc='upper right') #윗쪽의 상단에 라벨위치 설정
# plt.title('캘리포니아 Loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid() #격자표시 추가
# # font_path = 'C:Windows/Fonts/HYPost'
# # font_name = fm.FontProperties(fname=font_path).get_name()
# # plt.rf('font', family=font_name)
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.show()


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



