#1.data
import numpy as np
import time
import my_util

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
# (178, 13) (178,)

y = to_categorical(y)

print(y.shape)
# (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=99,
    stratify=y
)

scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# LSTM 입력은 3차원
# (samples, timesteps, features)

x_train = x_train.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)

print(x_train.shape)
print(x_test.shape)


#2.model
model = Sequential()

model.add(LSTM(64, input_shape=(13, 1)))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(3, activation='softmax'))

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
    patience=200,
    restore_best_weights=True
)

start_time = time.time()

batch_size = 6

history = model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=batch_size,
    verbose=2,
    validation_split=0.1,
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

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)

print("acc :", acc_score)
print("걸린시간 :", round(train_time, 2), "초")
print(y_test)

my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=99,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    test_loss=result,
    csv_file_path="./keras/model_history_log_v2.csv"
)