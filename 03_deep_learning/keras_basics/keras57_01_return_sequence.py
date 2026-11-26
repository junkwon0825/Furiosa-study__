import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ,ReduceLROnPlateau

#1.data 
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60] 
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70]) 

x = x.reshape(x.shape[0], x.shape[1], 1)
#2.model
model = Sequential()
model.add(LSTM(units=10, input_shape=(3,1), return_sequences=True))
model.add(LSTM(20,return_sequences=True))
model.add(LSTM(5))
model.add(Dense(1))

model.summary()

#3.compile,train
learning_rate = 0.001

model.compile(loss='mse',  optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'])

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

model.fit(
    x,
    y,
    epochs=1000,
    batch_size=4,
    validation_split=0.2,
    callbacks=[rlr, es]
)
#4.evaluate, predict
results = model.evaluate(x,y)
print('loss : ', results)

x_predict = np.array([50,60,70]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[50,60,70] 다음 예측의 결과: ', y_predict)











