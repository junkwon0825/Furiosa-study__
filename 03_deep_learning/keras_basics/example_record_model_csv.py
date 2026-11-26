import numpy as np
import time
import keras.my_util as my_util
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test =  np.array([8,9,10])
y_test =  np.array([8,9,10])

#2. 모델 구성
model = Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss = "mse",optimizer="adam")

batch_size = 4
start_time = time.time()

history = model.fit(x_train,y_train,epochs=500, batch_size=batch_size)

training_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:",loss)

result = model.predict(np.array([11]))
print("result:",result)

my_util.record_model_csv(
    model = model,
    data_shape=x_train.shape,
    batch_size = batch_size,
    history = history,
    training_time = training_time,
    test_loss = loss
)