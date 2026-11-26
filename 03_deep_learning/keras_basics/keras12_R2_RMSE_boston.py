#11-3 copy 내용
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np
import time
import my_util

#1.data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)
#-> 데이터 전처리 과정 모델에 데이터를 넣기 전에 데이터 정리하는 단계 들어갈 수 있음

#2.model
model = Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dense(30))
model.add(Dense(62))
model.add(Dense(48))
model.add(Dense(35))
model.add(Dense(15))
model.add(Dense(1))

#3.compile, train
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

batch_size = 20
history = model.fit(x_train,y_train, epochs=300, batch_size = 5)

train_time = time.time() - start_time
print("==========================================")

#4.evaluate, predict
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict)) #수식 추가하려면 np()안에
# return값을 써야지 반환이 됨 안쓰면 함수 정의가 안됨. rmse함수정의 하는방법
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)



my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2 = r2
)

#loss :  26.41722869873047








