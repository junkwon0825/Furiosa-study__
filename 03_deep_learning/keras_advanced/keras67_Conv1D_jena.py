#1.data
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


path = "./_data/kaggle_jena/"

dataset = pd.read_csv(
    path + 'jena_climate_2009_2016.csv',
    index_col=0
)

# 마지막 하루 실제 온도
y_cor = dataset[-144:]['T (degC)']

# 현재 하루 기상데이터 13개
x_data = dataset[:-288].drop(['T (degC)'], axis=1)

# 다음 하루 온도
y_data = dataset[144:-144]['T (degC)']


size_x = 144
size_y = 144


def split_x(dataset, size):
    aaa = []

    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i+size]
        aaa.append(subset)

    return np.array(aaa)


x = split_x(x_data, size_x)
y = split_x(y_data, size_y)

print("x :", x.shape)
print("y :", y.shape)

# x는 이미 Conv1D용 3차원
# (N, 144, 13)
# x reshape 필요 없음

# y : (N,144) → (N,144,1)
y = y.reshape(y.shape[0], y.shape[1], 1)


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    shuffle=False
)

print('x_train :', x_train.shape)
print('x_test :', x_test.shape)
print('y_train :', y_train.shape)
print('y_test :', y_test.shape)


# 마지막 하루 예측용
x_predict = dataset[-288:-144].drop(
    ['T (degC)'],
    axis=1
)

x_predict = np.array(x_predict)
y_cor = np.array(y_cor)

# Conv1D 입력
x_predict = x_predict.reshape(1, 144, 13)

# 정답 shape
y_cor = y_cor.reshape(1, 144, 1)

print('x_predict :', x_predict.shape)
print('y_cor :', y_cor.shape)


#2.model
model = Sequential()

model.add(
    Conv1D(
        64,
        kernel_size=3,
        input_shape=(144, 13),
        activation='relu',
        padding='same'
    )
)

model.add(
    Conv1D(
        32,
        kernel_size=3,
        activation='relu',
        padding='same'
    )
)

model.add(Dropout(0.2))

model.add(
    Conv1D(
        16,
        kernel_size=3,
        activation='relu',
        padding='same'
    )
)

model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(5, activation='relu'))

model.add(Dense(1))

model.summary()


#3.compile,train
learning_rate = 0.001

model.compile(
    loss='mse',
    optimizer=Adam(
        learning_rate=learning_rate
    )
)

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
    x_train,
    y_train,
    epochs=100,
    batch_size=500,
    validation_data=(x_test, y_test),
    callbacks=[rlr, es],
    shuffle=False
)


#4.evaluate,predict
results = model.evaluate(
    x_test,
    y_test
)

print('test loss :', results)

y_test_predict = model.predict(x_test)

rmse = root_mean_squared_error(
    y_test,
    y_test_predict
)

print('test RMSE :', rmse)


y_predict = model.predict(x_predict)

print('예측 shape :', y_predict.shape)
# (1, 144, 1)


results2 = model.evaluate(
    x_predict,
    y_cor
)

rmse2 = root_mean_squared_error(
    y_cor,
    y_predict
)

print('마지막 하루 loss :', results2)
print('마지막 하루 RMSE :', rmse2)


y_submit = y_predict.reshape(144,)

submission = pd.DataFrame({
    'T (degC)': y_submit
})

submission.to_csv(
    './submission.csv',
    index=False
)