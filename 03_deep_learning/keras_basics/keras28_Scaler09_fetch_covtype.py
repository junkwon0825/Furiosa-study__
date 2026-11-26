import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import time
import pandas as pd
import my_util
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. data
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(581012, 54) (581012,)



# print(np.unique(y)) #[1 2 3 4 5 6 7]
# print(np.unique(y,return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
# exit()

from sklearn.preprocessing import OneHotEncoder #onehotencoder가져왔으니까 정의
ohe = OneHotEncoder(sparse_output=False) # sparse->혼돈행렬 형태로 나옴 그래서 sparse_output=False
y = y.reshape(-1,1)
y = ohe.fit_transform(y)#벡터형태 데이터를 행렬형태 데이터로 변환해야함 (150,)-> (150,1) reshape
print(y.shape)

 # (581012, 7)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=78,
    stratify=y, #y데이터를 stratify하게 한다. -> 분류에서는 해주고 y기준으로 동일하게 잘림.
)
from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #train의 xmin,xmax학습 후 변환시킴 모두 0~1사이로
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  #(464809, 54) (116203, 54)
print(y_train.shape, y_test.shape)  #(464809, 8) (116203, 8)

#2.model
model = Sequential()
model.add(Dense(500, input_dim=54, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(7, activation='softmax')) 


#3.compile, train
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=200,
    restore_best_weights=True,
)
start_time = time.time()
batch_size=80000
history = model.fit(x_train, y_train, epochs=500, batch_size=4096, 
         verbose=1, validation_split=0.2, callbacks=[es],)
train_time = time.time() - start_time

#4.evaluate, predict
result = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_predict)
print("acc :  ", acc_score)
print("걸린시간 : ", round(train_time, 2), "초")
print(y_test)


my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=78,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    test_loss=result,
    csv_file_path="./keras/model_history_log_v2.csv"
)


#acc :   0.9043312134798586
#걸린시간 :  3077.57 초