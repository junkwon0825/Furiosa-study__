import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau


#1.data
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1) 
print(x.shape) #(7,3,1)

#2.model
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3,1)))
# model.add(SimpleRNN(30, input_shape=(3,1)))
model.add(SimpleRNN(units=10, input_length=3, input_dim=1))
#3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, ))
model.add(Dense(50, ))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.summary()

#3.compile,train
learning_rate = 0.005
# learning_rate = 0.001 디폴트
# learning_rate = 0.05
# learning_rate = 0.005
# learning_rate = 0.01
# learning_rate = 0.009
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

x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과: ', y_predict)









