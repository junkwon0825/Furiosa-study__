#1.data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import time
import my_util

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping


datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
# (581012, 54) (581012,)

ohe = OneHotEncoder(sparse_output=False)

y = y.reshape(-1, 1)
y = ohe.fit_transform(y)

print(y.shape)
# (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    shuffle=True,
    random_state=78,
    stratify=y
)

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
# (464809, 54) (116203, 54)

print(y_train.shape, y_test.shape)
# (464809, 7) (116203, 7)

# LSTM 입력은 3차원
# (samples, timesteps, features)

x_train = x_train.reshape(-1, 9, 6)
x_test = x_test.reshape(-1, 9, 6)

print(x_train.shape)
# (464809, 9, 6)

print(x_test.shape)
# (116203, 9, 6)


#2.model
model = Sequential()

model.add(LSTM(64, input_shape=(9, 6)))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(7, activation='softmax'))

model.summary()


#3.compile,train
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

start_time = time.time()

batch_size = 4096

history = model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)

train_time = time.time() - start_time


#4.evaluate,predict
result = model.evaluate(x_test, y_test)

print("loss :", result[0])
print("acc :", result[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)

print("accuracy_score :", acc_score)
print("걸린시간 :", round(train_time, 2), "초")

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