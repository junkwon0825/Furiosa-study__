import numpy as np
import time
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score


#1.data
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=5000,
    test_split=0.2
)

print(x_train.shape, y_train.shape)   # (8982,) (8982,)
print(x_test.shape, y_test.shape)     # (2246,) (2246,)

print(np.unique(y_train))
# 0 ~ 45
# 총 46개 클래스 -> 다중분류

print(type(x_train))
# <class 'numpy.ndarray'>

print(type(x_train[0]))
# <class 'list'>

print(len(x_train[0]), len(x_train[1]))

print("뉴스기사의 최대길이 :", max(len(i) for i in x_train)) #2376
print("뉴스기사의 최소길이 :", min(len(i) for i in x_train)) # 13
print("뉴스기사의 평균길이 :", sum(map(len, x_train)) / len(x_train)) #145.53

# Padding (전처리)
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

# print(x_train.shape)   # (8982, 100)
# print(x_test.shape)    # (2246, 100)

# print(y_train.shape)   # (8982,)
# print(y_test.shape)    # (2246,)

#y 원핫 인코딩
y_train = to_categorical(y_train, 46)
y_test = to_categorical(y_test, 46)

#2.model
model = Sequential()

model.add(Embedding(input_dim=5000, output_dim=100, input_length=200))
model.add(LSTM(64))
model.add(Dense(96, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(46, activation='softmax'))

# model.summary()


#3.compile,train
learning_rate = 0.001

# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=Adam(learning_rate=learning_rate),
#     metrics=['acc']
# )

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=learning_rate),
    metrics=['acc']
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    factor=0.5
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=300,
    restore_best_weights=True,
    verbose=1
)

start_time = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=1000,
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

# print(y_predict.shape)  # (2246, 46)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

# print(y_predict.shape)  # (2246,)
# print(y_test.shape) #(2246,)

acc_score = accuracy_score(y_test, y_predict)

print('accuracy_score :', acc_score)
print('걸린시간 :', round(end_time-start_time, 2), '초')