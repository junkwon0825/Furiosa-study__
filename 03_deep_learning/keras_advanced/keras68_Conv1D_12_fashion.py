#1.data
import numpy as np
import time

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)
# (60000, 28, 28) (60000,)

print(x_test.shape, y_test.shape)
# (10000, 28, 28) (10000,)


# scaling
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5


# RNN 입력은 3차원
# (samples, timesteps, features)

x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

print(x_train.shape)
# (60000, 28, 28)

print(x_test.shape)
# (10000, 28, 28)


ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(y_train.shape)
# (60000, 10)

print(y_test.shape)
# (10000, 10)


#2.model
model = Sequential()

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(8,8)))
model.add(Conv1D(10,2))
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

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
    patience=20,
    restore_best_weights=True,
    verbose=1
)

start_time = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=60,
    verbose=1,
    callbacks=[es],
    validation_split=0.2
)

end_time = time.time()


#4.evaluate,predict
print('===============model.evaluate==============')

loss = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)

print('accuracy_score :', acc_score)

print(
    '걸린시간 :',
    round(end_time - start_time, 2),
    '초'
)