import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.3,1.9],
            [9,8,7,6,5,4,3,2,1,0]
              ])
x = x.transpose()
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2.모델구성
model = Sequential()
model.add(Dense(10, input_dim=3)) 
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(1))


#3.컴파일,훈련 하는코드
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=395, batch_size=2)


#4.평가,예측
loss = model.evaluate(x,y)
print("loss : ", loss)

results = model.predict(np.array([[10, 1.3, 0]]))
print("[10, 1.3, 0]의 예측값: ", results)
