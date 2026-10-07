#1.data
import numpy as np
import time

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)
# (50000, 32, 32, 3) (50000, 1)

print(x_test.shape, y_test.shape)
# (10000, 32, 32, 3) (10000, 1)


# scaling
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5


# RNN 입력은 3차원
# (samples, timesteps, features)
x_train = x_train.reshape(-1, 32, 32 * 3)
x_test = x_test.reshape(-1, 32, 32 * 3)

print(x_train.shape)
# (50000, 32, 96)

print(x_test.shape)
# (10000, 32, 96)


ohe = OneHotEncoder(sparse_output=False)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(y_train.shape)
# (50000, 10)

print(y_test.shape)
# (10000, 10)


#2.model
model = Sequential()

model.add(
    SimpleRNN(
        256,
        input_shape=(32, 96),
        return_sequences=True
    )
)

model.add(SimpleRNN(128))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

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
    patience=30,
    restore_best_weights=True,
    verbose=1
)

start_time = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=300,
    batch_size=50,
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