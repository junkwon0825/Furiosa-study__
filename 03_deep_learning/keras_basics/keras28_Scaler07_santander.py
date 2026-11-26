#https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import numpy as np
import time
import pandas as pd
import my_util
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score



#1. data
path = "./_data/kaggle_santander/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sample_submission.csv", index_col=0)

x = train_csv.drop(['target'], axis=1) 
print(x.shape) 
y = train_csv['target']
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=90,
    stratify=y,
)

from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #train의 xmin,xmax학습 후 변환시킴 모두 0~1사이로
x_test = scaler.transform(x_test)

# print(np.unique(y_train, return_counts=True))
# print(np.unique(y_test, return_counts=True)) 


print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape)

#2. model
model = Sequential()
model.add(Dense(800, input_dim=200, activation='relu'))
model.add(Dense(600, activation='relu')) 
model.add(Dense(400)) 
model.add(Dense(200, activation='relu'))
model.add(Dense(100))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 


#3. compile , train
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'], 
              ) # 이진분류 loss도 저것만 씀 분류모델
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=300,
    restore_best_weights=True, 
)
start_time = time.time()
batch_size=9000
history = model.fit(x_train, y_train, epochs=100, batch_size = 9000,
                    verbose=1, validation_split=0.2,
                    callbacks=[es],
                    )

train_time = time.time() - start_time


#4. evaluate, predict
loss = model.evaluate(x_test,y_test)
result = model.evaluate(
    x_test,
    y_test,
    return_dict=True
)
# print("=====================================")
# print("loss : ", round(loss[0],4))
# print("acc :  ", round(loss[1],4))

y_predict = model.predict(x_test)

y_predict = np.round(y_predict) #반올림 꼭 해줘야함 test랑 같게
acc_score = accuracy_score(y_test, y_predict)
print("acc :  ", acc_score)

r2 = r2_score(y_test, y_predict)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print(f"RMSE : {rmse : .2f}")

test_csv = scaler.transform(test_csv)
y_submit = model.predict(test_csv)
submission['count'] = y_submit

submission.to_csv(path + "submit/" + "submit_0910_1830.csv", index=True)

my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=78,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    test_loss=result,
    csv_file_path="./keras/model_history_log_v2.csv"
)
