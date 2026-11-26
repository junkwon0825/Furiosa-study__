from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1.데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])
# 데이터 1:1 매칭이 아니라 전체가 전체로 대응

#2.모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(9))
model.add(Dense(4))
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))


#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=379, batch_size=2)
#데이터 자르는 과정 -> batch 3은 3개 데이터크기로 쪼갰다는 뜻 6/3=2

#4.평가, 예측
loss = model.evaluate(x,y)
print("loss : ", loss)
# result = model.predict(np.array([1,2,3,4,5]))
# print("7의 예측값 : ", result)