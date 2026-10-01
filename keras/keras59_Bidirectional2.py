#55-2copy
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1.data 
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60] 
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0], x.shape[1], 1) 

#2.model
model = Sequential()
model.add(Bidirectional(LSTM(30), input_shape=(3,1))) # simplernn을 양쪽으로 작동해라
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, ))
model.add(Dense(50, ))
model.add(Dense(30, ))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

#3.compile,train
learning_rate = 0.008

model.compile(loss='mse',  optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'])

model.fit(x,y,epochs=1000)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode= 'auto',
    patience=200,
    verbose=1,
    factor=0.5,
)
es = EarlyStopping(
    monitor='val_loss',          # validation loss 감시
    mode='auto',                  # val_loss는 작을수록 좋음
    patience=500,                 # 10 epoch 동안 개선 없으면 중단
    restore_best_weights=True,   # 가장 좋았던 weight로 복구
    verbose=1
)

#4.evaluate, predict
results = model.evaluate(x,y)
print('loss : ', results)

x_predict = np.array([50,60,70]).reshape(1,3,1)        #80 맞춰보야요

y_predict = model.predict(x_predict)

print('[50,60,70]의 결과: ', y_predict)


















