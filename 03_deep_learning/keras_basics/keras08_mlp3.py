import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1.data
x = np.array([range(10), range(21,31), range(201,211)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
               [10,9,8,7,6,5,4,3,2,1,]]).transpose()
print(x.shape, y.shape) #(10,3) , (10,2)


#2.model
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(11))
model.add(Dense(5))
model.add(Dense(2))

#3.compile,train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=440, batch_size=3)

#4.test,predict
loss = model.evaluate(x,y)
print("loss : ", loss)

results = model.predict(np.array([[10, 31, 211]]))
print("[10, 31, 211]의 예측값 : ", results)

