import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


a = np.array(range(1,11)) ## 벡터 형태 (10,)
size = 5  # timestep size

def split_xy(dataset, size):
    x = []
    y = []

    for i in range(len(dataset) - size):
        x_subset = dataset[i : i+size]
        y_subset = dataset[i+size]

        x.append(x_subset)
        y.append(y_subset)

    return np.array(x), np.array(y)
x , y = split_xy(a,size)
x = x.reshape(x.shape[0], x.shape[1], 1) 

print(x)
print(x.shape) #(5,5,1)
#2.model
model = Sequential()
model.add(GRU(30, input_shape=(5,1)))
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

x_predict = np.array([6,7,8,9,10]).reshape(1,5,1)

y_predict = model.predict(x_predict)

print('[6,7,8,9,10]의 결과: ', y_predict)


















