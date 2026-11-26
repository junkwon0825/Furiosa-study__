import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import time
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


#1.data
docs = [
    '너무 재미있다', '참 최고예요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미 없다', '참 재밌네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)

print(token.word_index)

x = token.texts_to_sequences(docs)

print(x)


### 패딩 ###
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_x = pad_sequences(
    x,
    maxlen=5,
    padding='pre',
    truncating='post'
)

print(padded_x)
print(padded_x.shape)      # (15, 5)


### 원핫인코딩 ###
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

x = padded_x.reshape(-1, 1)
print(x.shape)             # (75, 1)

x = ohe.fit_transform(x)
print(x.shape)             # (75, 32)

x = x.reshape(15, 5, 32)
print(x.shape)             # (15, 5, 32)

y = labels


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=78,
    stratify=y,
)

print(x_train.shape)       # (12, 5, 32)
print(x_test.shape)        # (3, 5, 32)


#2.model
model = Sequential()

model.add(LSTM(5, input_shape=(5, 32)))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3.compile,train
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=500,
    restore_best_weights=True
)

start_time = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=5,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)

train_time = time.time() - start_time


#4.evaluate,predict
loss = model.evaluate(x_test, y_test)

print("=====================================")
print("loss :", round(loss[0], 4))
print("acc  :", round(loss[1], 4))


### 새로운 문장 예측 ###
x_predict = ["개똥이 잘생겼다"]

x_predict = token.texts_to_sequences(x_predict)

print(x_predict)

x_predict = pad_sequences(
    x_predict,
    maxlen=5,
    padding='pre',
    truncating='post'
)

print(x_predict)
print(x_predict.shape)     # (1, 5)


### 예측 데이터도 똑같이 원핫인코딩 ###
x_predict = x_predict.reshape(-1, 1)

x_predict = ohe.transform(x_predict)

print(x_predict.shape)     # (5, 32)

x_predict = x_predict.reshape(1, 5, 32)

print(x_predict.shape)     # (1, 5, 32)


x_pred = model.predict(x_predict)

print("예측 확률 :", x_pred)
print("예측 결과 :", np.round(x_pred))


### 테스트 데이터 정확도 ###
y_predict = model.predict(x_test)

y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)

print("acc :", acc_score)

r2 = r2_score(y_test, y_predict)