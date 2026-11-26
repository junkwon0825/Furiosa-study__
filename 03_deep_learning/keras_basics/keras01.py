import tensorflow as tf
print(tf.__version__)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=150) #학습하는것 x,y데이터 많을수록 좋음

#4. 평가 예측
result = model.predict(np.array([4]))
print("4의 예측값 : ", result)

#결과
