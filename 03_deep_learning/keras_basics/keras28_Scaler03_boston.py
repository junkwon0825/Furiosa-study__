from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
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

#3.compile, train
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(#class
    monitor='val_loss',
    mode = 'auto', #뭔지 헷갈릴때는 auto 잡기 loss는 min이긴함.
    patience=10, #참는다 인내심 최소가 더 나오는지 기다리는거
    restore_best_weights=True, #이거 안쓰면 10번째 뒤에게 채택됨 
)

start_time = time.time()
batch_size=30
history = model.fit(x_train,y_train, epochs=500, batch_size = 30,
                    verbose=1, validation_split=0.15,
                    callbacks=[es], #2개이상은 리스트. es를 리스트형태로 받아들임.
                    )

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
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# print("====================== history ===============================")
# plt.figure(figsize=(9,6)) #그냥 그림판 자체 크기 사이즈
# plt.plot(history.history['loss'][30:], c='red', label='loss') #y값만 넣으면 x디폴트는 시간순으로 그려줌
# plt.plot(history.history['val_loss'][30:], c='blue', label='val_loss')
# plt.legend(loc='upper right')
# plt.title('boston Loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid() 
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False
# plt.show()
#loss :  26.41722869873047


my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=78,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    # test_loss=result,
    csv_file_path="./keras/model_history_log_v2.csv"
)
