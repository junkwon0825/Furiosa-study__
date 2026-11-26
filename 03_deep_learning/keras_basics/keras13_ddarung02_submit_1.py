#https://dacon.io/competitions/open/235576/leaderboard

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import time
import my_util

#1.data
path = "./_data/ddarung/" #상대경로
# path = "c:/study/_data/ddarung/" #절대경로
# path = "c:\study\_data\ddarung/" #슬래시 역슬래시 상관없어
# path = "c://study//_data//ddarung/" #//두개도 상관없음
# path = "c:\\study\\_data\\ddarung/" #\\두개도 상관없어



train_csv = pd.read_csv(path + "train.csv", index_col=0) #문자열 내용 합치기
#pd.read_csv가 읽은 데이터를 train_csv가 가져온다.
print(train_csv) # id열 미포함->[1459 rows x 10 columns] index_col=0때문
#column명 자체는 데이터가 아니다.

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) #[715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission) #[715 rows x 1 columns]

print(train_csv.columns) #컬럼명들이 나옴
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')
print(train_csv.info()) #데이터 정보를 가져옴

# exit()#-> 코드 중단점.
############################# 결측치 처리 1.삭제 #############################

train_csv = train_csv.dropna()
print(train_csv) #[1328 rows x 10 columns] 130개정도 사라짐 ㅋㅋ

# train_csv를 x와 y로 분리
x = train_csv.drop(['count'], axis=1) #축 행:0 열:1 count라는 열 삭제
print(x) #[1328 rows x 9 columns]

y = train_csv['count']
print(y.shape) #(1328,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.82,
    shuffle=True,
    random_state=91
)

############################## submit 물밑작업 #############################
print(test_csv.info()) #-> 결측치 많음 그래서 결측치 처리

############################# 결측치 처리 2.평균값 넣기 #############################
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())
print(test_csv.shape)

# exit()
#2.model
model = Sequential()
model.add(Dense(512, input_dim=9))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3.compile,train
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

batch_size = 3
history = model.fit(x_train,y_train, epochs=380, batch_size = 3)

train_time = time.time() - start_time

#4.evaluate, predict 평가,예측ㄱㄱ
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

############## submission.csv 만들기 // count 컬럼에 값 넣어준다. ######
print(submission)
y_submit = model.predict(test_csv)  #predict도 여러번 해도됨
submission['count'] = y_submit #submission에 count라는 열안에 y_submit넣음

submission.to_csv(path + "submit/" + "submit_0904_1309.csv", index=True)
# 파일 불러올때 read_csv, 저장할때 csv
my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 79,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2 = r2
)

"""
하이퍼 파리미터 튜닝 -> 숫자값 작성 명세
random_state
train_size
레이어 깊이
노드의 개수
epoch
batch_size

"""

"""

1차시도 
random : 79
train_size=0.8
epochs=380
batch_size = 5

"""