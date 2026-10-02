#1.data
import numpy as np
import time
import pandas as pd
import my_util

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score


path = "./_data/kaggle_santander/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sample_submission.csv", index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)       # (200000, 200)
print(y.shape)       # (200000,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=90,
    stratify=y
)

scaler = RobustScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
# (160000, 200) (40000, 200)

print(y_train.shape, y_test.shape)
# (160000,) (40000,)

# LSTM 입력 3차원
# (samples, timesteps, features)
x_train = x_train.reshape(-1, 20, 10)
x_test = x_test.reshape(-1, 20, 10)

print(x_train.shape)
# (160000, 20, 10)

print(x_test.shape)
# (40000, 20, 10)


#2.model
model = Sequential()

model.add(LSTM(64, input_shape=(20, 10)))

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
    patience=30,
    restore_best_weights=True
)

start_time = time.time()

batch_size = 9000

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)

train_time = time.time() - start_time


#4.evaluate,predict
result = model.evaluate(
    x_test,
    y_test,
    return_dict=True
)

print("loss :", result['loss'])
print("acc :", result['acc'])

y_predict = model.predict(x_test)

# accuracy 계산용
y_predict_acc = np.round(y_predict).reshape(-1)

acc_score = accuracy_score(
    y_test,
    y_predict_acc
)

print("accuracy_score :", acc_score)


# Kaggle 제출 데이터
test_csv = scaler.transform(test_csv)

# LSTM shape
test_csv = test_csv.reshape(-1, 20, 10)

print(test_csv.shape)

# 제출할 때는 확률 그대로 사용
y_submit = model.predict(test_csv).reshape(-1)

submission['target'] = y_submit

submission.to_csv(
    path + "submit/" + "submit_LSTM.csv",
    index=True
)

my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=90,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    test_loss=result,
    csv_file_path="./keras/model_history_log_v2.csv"
)