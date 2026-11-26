import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1.data
x = np.array(range(10)) #[0 1 2 3 4 5 6 7 8 9]
print(x)
x = np.array(range(1,10)) #[1 2 3 4 5 6 7 8 9]
print(x)
x = np.array(range(1,11)) #[1 2 3 4 5 6 7 8 9 10]
print(x)

x = np.array([range(10), range(21, 31,), range(201,211)]).T #(3,10)행렬 
# 전치시키고 싶으면 그냥 라인뒤에 .T 붙혀도 상관 없음 
print(x.shape)

y = np.array(range(1,11))
print(y.shape) #(10,)


#2.model
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(12))
model.add(Dense(5))
model.add(Dense(1))

#3.compile,train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=385, batch_size=2)

#4.test,predict
loss = model.evaluate(x,y)
print("loss : ", loss)

results = model.predict(np.array([[10,31,211]]))
print("[10,31,311]의 예측값 : ", results)

#[10,31,311]의 예측값 :  [[11.000796]]