import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1.data
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
               [10,9,8,7,6,5,4,3,2,1,],
               [9,8,7,6,5,4,3,2,1,0]]).transpose()
print(x.shape, y.shape) #(10,)  (10,3) 가능한 모델 성능보장x

#2.model
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))

#3.compile,train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=380, batch_size=3)

#4.test,predict
loss = model.evaluate(x,y)
print("loss : ", loss)

results = model.predict(np.array([[10]]))
print("[10]의 예측값 : ", results)

#[10]의 예측값 :  [[ 1.1000000e+01  6.8843365e-06 -9.9999577e-01]]