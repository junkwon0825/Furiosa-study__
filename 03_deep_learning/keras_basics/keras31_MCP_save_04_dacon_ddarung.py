
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
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
from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #train의 xmin,xmax학습 후 변환시킴 모두 0~1사이로
x_test = scaler.transform(x_test)

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

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1,)


################# mcp 세이브 파일명 만들기 시작 #####################
import datetime
date = datetime.datetime.now() #현재 시간반환

print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") #0914_1148
print(date)
print(type(date)) #<class 'str'>


path = './_save/keras33/'
filename = '-{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k33_", date, filename]) #문자없이 빈공간. + join 뒤에 붙힌다. + 
#'./_save/keras30/' + "k30_" + #0914_1148 + '530-0.001.keras'

################# mcp 세이브 파일명 만들기 끝 #####################

# exit()
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=filepath, verbose=1,)
start_time = time.time()

batch_size=40
history = model.fit(x_train,y_train, epochs=500, batch_size = 40,
                    callbacks=[es,mcp],
                    verbose=1, validation_split=0.15)

train_time = time.time() - start_time


print("====================================================")
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
test_csv = scaler.transform(test_csv)
y_submit = model.predict(test_csv)
submission['count'] = y_submit

submission.to_csv(path + "submission_result.csv", index=True)


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
