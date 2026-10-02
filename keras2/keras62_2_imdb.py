from tensorflow.keras.datasets import imdb
import numpy as np
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score


#1.data
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=5000
)

print(x_train.shape, y_train.shape)   # (25000,) (25000,)
print(x_test.shape, y_test.shape)     # (25000,) (25000,)

print(np.unique(y_train))             # [0 1] -> 이진분류

print(type(x_train))
print(type(x_train[0]))

print(len(x_train[0]), len(x_train[1]))

print("리뷰의 최대길이 :", max(len(i) for i in x_train))
print("리뷰의 최소길이 :", min(len(i) for i in x_train))
print("리뷰의 평균길이 :", sum(map(len, x_train)) / len(x_train))

x_train = pad_sequences(
    x_train,
    maxlen=200,
    padding='post',
    truncating='pre'
)

x_test = pad_sequences(
    x_test,
    maxlen=200,
    padding='post',
    truncating='pre'
)

print(x_train.shape)   # (25000, 200)
print(x_test.shape)    # (25000, 200)

print(y_train.shape)   # (25000,)
print(y_test.shape)    # (25000,)


#2.model
model = Sequential()

model.add(Embedding(input_dim=5000, output_dim=100, input_length=200))
model.add(LSTM(64))

model.add(Dense(96, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()


#3.compile,train
learning_rate = 0.001

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=learning_rate),
    metrics=['acc']
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
    patience=200,
    restore_best_weights=True,
    verbose=1
)

start_time = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=128,
    verbose=1,
    callbacks=[es, rlr],
    validation_split=0.2
)

end_time = time.time()


#4.evaluate,predict
print('===============model.evaluate==============')

loss = model.evaluate(x_test, y_test, verbose=1)

print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)

print(y_predict.shape)       # (25000, 1)

y_predict = np.round(y_predict).reshape(-1)

print(y_predict.shape)       # (25000,)
print(y_test.shape)          # (25000,)

acc_score = accuracy_score(y_test, y_predict)

print('accuracy_score :', acc_score)
print('걸린시간 :', round(end_time-start_time, 2), '초')


#Embedding + LSTM/GRU
#Embedding + Bidirectional
#Embedding + Flatten + DNN
