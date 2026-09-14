from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
import my_util
import numpy as np
#1.data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #train의 xmin,xmax학습 후 변환시킴 모두 0~1사이로
x_test = scaler.transform(x_test)

#2.model
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(30))
model.add(Dense(62))
model.add(Dense(48))
model.add(Dense(23))
model.add(Dense(1))

#3.compile,train
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.models import load_model

path = './_save/keras32/'

model = load_model( path + 'k32_0914_1355-0043-24.5795.keras')

#4.evaluate, predict
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


# my_util.record_model_csv(
#     model=model,
#     data_shape=x_train.shape,
#     random_num=78,
#     batch_size=batch_size,
#     history=history,
#     training_time=train_time,
#     # test_loss=result,
#     csv_file_path="./keras/model_history_log_v2.csv"
# )
