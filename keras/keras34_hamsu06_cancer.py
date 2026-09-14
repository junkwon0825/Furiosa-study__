import numpy as np
import time
import pandas as pd
import my_util
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer #유방암관련 데이터셋 불러오기


#1. data
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets['data'] # 딕셔너리 형태가 베이스 
# x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(569, 30) (569,)
print(type(x)) #<class 'numpy.ndarray'>

print(y) #0과1의 개수가 몇개인지 찾아보기, numpy
print(np.unique(y)) #[0 1] -> y의 데이터 중에 독특한놈 있냐. 0,1만 있음
print(np.unique(y,return_counts=True)) # 개수를 return 할래? ㅇㅇ #(array([0, 1]), array([212, 357]))
#0과 1의 개수가 몇개인지 찾아보기, pandas
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212

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

print(np.unique(y_train, return_counts=True)) #(array([0, 1]), array([167, 288]))
print(np.unique(y_test, return_counts=True)) #(array([0, 1]), array([45, 69]))

##-> 잘한걸까 못한걸까. test데이터는 평가만 하는거니까 불균형해도 상관 x
## train 데이터는 불균형하면 안됨.

print(x_train.shape, x_test.shape)  #(398, 30) (171, 30)
print(y_train.shape, y_test.shape)  #(398,) (171,)

#2. model
# model = Sequential()
# model.add(Dense(60, input_dim=30, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(50)) # 아무것도 안쓰면 디폴트 y=wx+b linear
# model.add(Dense(40, activation='relu')) #relu
# model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid')) # 마지막은 무조건 sigmoid 함수 넣어야함

#함수형 모델
input1 = Input(shape=(30,))
dense1 = Dense(60, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(50)(drop1) 
dense3 = Dense(40, activation='relu')(dense2)
dense4 = Dense(30, activation='relu')(dense3)
drop2 = Dropout(0.2)(dense4)
dense5 = Dense(20, activation='relu')(drop2)
dense6 = Dense(10, activation='relu')(dense5)
drop3 = Dropout(0.2)(dense6)
output1 = Dense(1, activation='sigmoid')(drop3)
model = Model(inputs=input1, outputs=output1)

#3. compile , train
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1,)


################# mcp 세이브 파일명 만들기 시작 #####################
import datetime
date = datetime.datetime.now() #현재 시간반환

print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") #0914_1148
print(date)
print(type(date)) #<class 'str'>


path = './_save/keras35/'
filename = '-{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k35_", date, filename]) #문자없이 빈공간. + join 뒤에 붙힌다. + 
#'./_save/keras30/' + "k30_" + #0914_1148 + '530-0.001.keras'

################# mcp 세이브 파일명 만들기 끝 #####################

# exit()
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=filepath, verbose=1,)
start_time = time.time()

batch_size=40
history = model.fit(x_train,y_train, epochs=500, batch_size = 40,
                    callbacks=[es,],
                    verbose=1, validation_split=0.15)

train_time = time.time() - start_time


print("====================================================")

# 4. evaluate, predict

result = model.evaluate(x_test, y_test, return_dict=True)

y_predict = model.predict(x_test)

# 확률 → 0 또는 1
y_predict = (y_predict > 0.5).astype(int).ravel()

acc_score = accuracy_score(y_test, y_predict)

print("accuracy :", acc_score)

# CSV 기록
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
# acc :   0.9912280701754386
# RMSE :  0.09