import numpy as np
import pandas as pd
import time
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM,GRU,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

#1.data

path = "./_data/kaggle_jena/"

dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

print(dataset.shape)
print(dataset.head())
print(dataset.columns)


# 마지막 하루의 실제 풍향
# 마지막에 예측값과 비교하기 위한 정답
y_cor = dataset[-144:]['T (degC)']


# x : 현재 하루의 기상데이터 13개
# 마지막 288개는 제외
# 풍향은 x에서 제거
x_data = dataset[:-288].drop(['T (degC)'], axis=1)


# y : x보다 144칸 뒤의 풍향
# 즉 다음 하루 풍향
y_data = dataset[144:-144]['T (degC)']


print('x_data :', x_data.shape)
print('y_data :', y_data.shape)
print('y_cor :', y_cor.shape)


size_x = 144
size_y = 144


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i+size]
        aaa.append(subset)
    return np.array(aaa)


start_time = time.time()

x = split_x(x_data, size_x)
y = split_x(y_data, size_y)

end_time = time.time()

print('split 시간 :', end_time - start_time)

print('x shape :', x.shape)
print('y shape :', y.shape)


# y : (N,144) → (N,144,1)
y = y.reshape(y.shape[0], y.shape[1], 1)

# print('reshape 후 y :', y.shape)

# train / test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=False
)

print('x_train :', x_train.shape)
print('x_test :', x_test.shape)
print('y_train :', y_train.shape)
print('y_test :', y_test.shape)


# 마지막 하루를 예측하기 위한 전날 데이터
x_predict = dataset[-288:-144].drop(['T (degC)'], axis=1)

x_predict = np.array(x_predict)
y_cor = np.array(y_cor)

x_predict = x_predict.reshape(1, 144, 13)
y_cor = y_cor.reshape(1, 144, 1)

print('x_predict :', x_predict.shape)
print('y_cor :', y_cor.shape)



#2.model

model = Sequential()
model.add(Bidirectional(LSTM(10, input_shape=(144,13),return_sequences=True)))
model.add(Bidirectional(LSTM(30,return_sequences=True)))
model.add(Bidirectional(LSTM(50,return_sequences=True)))
model.add(Bidirectional(LSTM(50,return_sequences=True)))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))



#3.compile,train

learning_rate = 0.001

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=10,
    verbose=1,
    factor=0.5
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=700,
    validation_data=(x_test, y_test),
    callbacks=[rlr, es],
    shuffle=False
)


#4.evaluate,predict

results = model.evaluate(x_test, y_test)
print('test loss :', results)

y_test_predict = model.predict(x_test)
rmse = root_mean_squared_error(y_test, y_test_predict)

print('test RMSE :', rmse)


y_predict = model.predict(x_predict)

print('예측 shape :', y_predict.shape)
print('예측 풍향 :')
print(y_predict)

results2 = model.evaluate(x_predict, y_cor)
rmse2 = root_mean_squared_error(y_cor, y_predict)

print('마지막 하루 loss :', results2)
print('마지막 하루 RMSE :', rmse2)
print('실제 풍향 :')
print(y_cor)



y_predict = model.predict(x_predict)

y_submit = y_predict.reshape(144,)

submission = pd.DataFrame({'T (degC)': y_submit})

submission.to_csv('./submission.csv', index=False)