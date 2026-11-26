import numpy as np
import time
import pandas as pd
import my_util
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


#1.data
datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) 


from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)
 #(178, 3)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=99,
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

print(x_train.shape, x_test.shape)  #(124, 13) (54, 13)
print(y_train.shape, y_test.shape)  #((124, 3) (54, 3)

#2.model
model = Sequential()
model.add(Dense(100, input_dim=13, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='softmax')) 


#3.compile, train
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=200,
    restore_best_weights=True,
)
start_time = time.time()
batch_size=6
history = model.fit(x_train, y_train, epochs=1000, batch_size=6, 
         verbose=2, validation_split=0.1, callbacks=[es],)
train_time = time.time() - start_time

#4.evaluate, predict
result = model.evaluate(x_test,y_test, return_dict=True)
print(result) # 그냥 loss적으면 metrics들어가있어서 2개나옴
# print('acc : ', round(result[1],2))

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



## 목표값 acc = 0.95
#acc :   0.9444444444444444 걸린시간 :  192.6 초
#acc :   0.9722222222222222 걸린시간 :  73.68 초 