import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
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
x_train = x_train.reshape(-1, 2, 5, 1)
x_test  = x_test.reshape(-1, 2, 5, 1)

#2.model
model = Sequential()
model.add(Conv2D(64,(1,2),input_shape=(2,5,1), activation='relu', padding='same')) #(26,26,64)
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
model.add(Dense(1)) # (10,)


#3.compile,train
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(#class
    monitor='val_loss',
    mode = 'auto', #뭔지 헷갈릴때는 auto 잡기 loss는 min이긴함.
    patience=100, #참는다 인내심 최소가 더 나오는지 기다리는거
    restore_best_weights=True, #이거 안쓰면 10번째 뒤에게 채택됨 
)
start_time = time.time()
batch_size=30
history = model.fit(x_train,y_train, epochs=500, batch_size = 30,
                    verbose=1, validation_split=0.15,
                    callbacks=[es], #2개이상은 리스트. es를 리스트형태로 받아들임.
                    )

train_time = time.time() - start_time
print("================================================")
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








