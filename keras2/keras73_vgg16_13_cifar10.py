import numpy as np
import time
import tensorflow as tf
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.datasets import cifar10

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)


#1.data

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)
# (50000, 32, 32, 3) (50000, 1)

print(x_test.shape, y_test.shape)
# (10000, 32, 32, 3) (10000, 1)


# VGG16 전용 전처리
x_train = preprocess_input(x_train.astype(np.float32))
x_test = preprocess_input(x_test.astype(np.float32))

print(x_train.shape)
# (50000, 32, 32, 3)

print(x_test.shape)
# (10000, 32, 32, 3)


ohe = OneHotEncoder(sparse_output=False)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(y_train.shape)
# (50000, 10)

print(y_test.shape)
# (10000, 10)


#2.model

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


#3.compile,train

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1)

start_time = time.time()

history = model.fit(x_train, y_train, epochs=300, batch_size=50, verbose=1, callbacks=[es], validation_split=0.2)

end_time = time.time()


#4.evaluate,predict

print('===============model.evaluate==============')

loss = model.evaluate(x_test, y_test, verbose=1)

print('loss :', loss[0])
print('acc :', loss[1])


y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)

print('accuracy_score :', acc_score)
print('걸린시간 :', round(end_time - start_time, 2), '초')