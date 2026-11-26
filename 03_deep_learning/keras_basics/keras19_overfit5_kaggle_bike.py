#https://www.kaggle.com/competitions/bike-sharing-demand/rules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import my_util
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


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


#2. model
model = Sequential()
model.add(Dense(512, input_dim=8, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1)) #마지막층은 relu 안하는게 좋음 마지막층은 default가 미니어
#3. compile, train
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

batch_size = 15
history = model.fit(x_train,y_train, epochs=345, batch_size = 15,
                    verbose=1, validation_split=0.2)

train_time = time.time() - start_time

#4. evaluate, predict
loss = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_submit = model.predict(test_csv)
submission['count'] = y_submit

submission.to_csv(path + "submit/" + "submit_0907_1430.csv", index=True)

print("====================== history ===============================")
plt.figure(figsize=(9,6)) #그냥 그림판 자체 크기 사이즈
plt.plot(history.history['loss'][10:], c='red', label='loss') #y값만 넣으면 x디폴트는 시간순으로 그려줌
plt.plot(history.history['val_loss'][10:], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('bike Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid() 
plt.legend()
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.show()
plt.show()

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 78,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2 = r2
)