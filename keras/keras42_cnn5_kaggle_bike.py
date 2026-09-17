#https://www.kaggle.com/competitions/bike-sharing-demand/rules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import my_util
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping

#1. data
path = "./_data/kaggle_bike/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

#print(train_csv.columns) #컬럼명들이 나옴
#print(train_csv.info()) #데이터 정보를 가져옴 -> 결측치 없음을 확인 바로진행
#print(train_csv.describe()) #데이터 정보 자세히 묘사

############ 결측치 확인 #############
print(train_csv.isna().sum()) #->결측치 있는지 확인하는 코드 이게 더 편해보이네?

x = train_csv.drop(['casual','registered','count'], axis=1) #축 행:0 열:1 count라는 열 삭제
print(x) #[10886 rows x 8 columns]

y = train_csv['count']
print(y, y.shape)  #(10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=78
)
from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #train의 xmin,xmax학습 후 변환시킴 모두 0~1사이로
x_test = scaler.transform(x_test)
print(x_test.shape,x_train.shape,y_test.shape,y_train.shape) #(2178, 8) (8708, 8) (2178,) (8708,)
x_train = x_train.reshape(-1, 4, 2, 1)
x_test  = x_test.reshape(-1, 4, 2, 1)
#2. model
model = Sequential()
model.add(Conv2D(64,(2,1),input_shape=(4, 2, 1), activation='relu', padding='same')) #(26,26,64)
model.add(Conv2D(filters=32, kernel_size=(2,1), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(30,(2,1),activation='relu', padding='same')) #(20,20,16)
model.add(Conv2D(30,(2,1),activation='relu', padding='same')) #(20,20,16)
model.add(Conv2D(30,(2,1),activation='relu', padding='same')) #(21,21,16)

model.add(Flatten()) #2차원변경
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(1)) # (10,)
#3. compile, train
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(#class
    monitor='val_loss',
    mode = 'auto', #뭔지 헷갈릴때는 auto 잡기 loss는 min이긴함.
    patience=50, #참는다 인내심 최소가 더 나오는지 기다리는거
    restore_best_weights=True, #이거 안쓰면 10번째 뒤에게 채택됨 
)
start_time = time.time()
batch_size=55
history = model.fit(x_train,y_train, epochs=1000, batch_size = 55,
                    verbose=1, validation_split=0.15,
                    callbacks=[es], #2개이상은 리스트. es를 리스트형태로 받아들임.
                    )

train_time = time.time() - start_time
#4. evaluate, predict
loss = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print(f"RMSE : {rmse : .2f}")

y_submit = model.predict(test_csv)
# test_csv = scaler.transform(test_csv)
submission['count'] = y_submit

submission.to_csv(path + "submit/" + "submit_0911_1830.csv", index=True)


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

# RMSE : 147.91 -> 146.42 RMSE :  146.61