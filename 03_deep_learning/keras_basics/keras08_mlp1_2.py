import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1.데이터 {shape확인)
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])
x = x.T # x = x.transpose() 행열위치 잘못썼을때 바꾸려면 사용
# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5])

print(x.shape) #(5,2)행렬 <-> (5,)벡터 input_dim ->1
print(y.shape) #(5,)

#2.모델구성
model = Sequential()
model.add(Dense(5, input_dim=2)) #(행5,열2) 2-5-7-3-1트리
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))


#3.컴파일,훈련 하는코드
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=490, batch_size=2)


#4.평가,예측
loss = model.evaluate(x,y)
print("loss : ", loss)

results = model.predict(np.array([[6,11]])) #predict에도 행무시 열우선 적용 (1,2)
print("[6,11]의 예측값: ", results)

