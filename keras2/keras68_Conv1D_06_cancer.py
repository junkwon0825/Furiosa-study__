import numpy as np
import time
import pandas as pd
import my_util

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import RobustScaler


#1.data
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (569, 30) (569,)
print(np.unique(y, return_counts=True))
# 0 : 212개
# 1 : 357개

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=78,
    stratify=y
)

scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
# (455, 30) (114, 30)

print(y_train.shape, y_test.shape)
# (455,) (114,)

# LSTM 입력은 3차원
# (samples, timesteps, features)
x_train = x_train.reshape(-1, 30, 1)
x_test = x_test.reshape(-1, 30, 1)

print(x_train.shape)
# (455, 30, 1)

print(x_test.shape)
# (114, 30, 1)


#2.model
model = Sequential()

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(30,1)))
model.add(Conv1D(10,2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()


#3.compile,train
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True
)

start_time = time.time()

batch_size = 55

history = model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.15,
    callbacks=[es]
)

train_time = time.time() - start_time


#4.evaluate,predict
result = model.evaluate(
    x_test,
    y_test,
    return_dict=True
)

print(result)

print("test loss :", result['loss'])
print("test acc :", result['acc'])

y_predict = model.predict(x_test).ravel()

print("확률값 :", y_predict[:10])

y_predict = (y_predict > 0.5).astype(int)

print("분류 결과 :", y_predict[:10])

acc_score = accuracy_score(y_test, y_predict)

print("accuracy_score :", acc_score)

print(
    "걸린시간 :",
    round(train_time, 2),
    "초"
)

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