import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
import time
from sklearn.metrics import accuracy_score, r2_score,mean_squared_error,mean_absolute_error
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
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0]) #(15,)

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)


print(x)
#[[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14],
#  [15], [16], [17, 18], [19, 20], [21], [2, 22, 23], [1, 24],
#  [25, 26], [27, 28], [29, 30, 31]]  ### 길이가 다 다르면 어떻게 만들까
# 언어의 길이가 다 다르기 때문에 제일 긴 애를 기준으로 빈자리를 0으로 채운다.

### 패딩 #####
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(
                x,
                maxlen=5,
                padding='pre',
                truncating='post'
)

print(padded_x)
print(padded_x.shape) #(15, 5)
y = labels
padded_x = padded_x.reshape(padded_x.shape[0], padded_x.shape[1], 1)
x_train, x_test, y_train, y_test = train_test_split(
    padded_x, y,
    train_size=0.8,
    shuffle=True,
    random_state=78,
    stratify=y,
)


#2. model
model = Sequential()
model.add(LSTM(30, input_shape=(5,1)))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

#3. compile , train
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'], 
              ) # 이진분류 loss도 저것만 씀 분류모델
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=300,
    restore_best_weights=True, 
)
start_time = time.time()
history = model.fit(x_train, y_train, epochs=1000, batch_size = 10,
                    verbose=1, validation_split=0.2,
                    callbacks=[es],
                    )

train_time = time.time() - start_time


#4. evaluate, predict
loss = model.evaluate(x_test,y_test)
print("=====================================")
print("loss : ", round(loss[0],4))
print("acc :  ", round(loss[1],4))

x_predict = ["개똥이 잘생겼다"]

x_predict = token.texts_to_sequences(x_predict)

x_predict = pad_sequences(
    x_predict,
    maxlen=5,
    padding='pre',
    truncating='post'
)
x_pred = model.predict(x_predict)
print(np.round(x_pred))

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)
print("acc :  ", acc_score)

r2 = r2_score(y_test, y_predict)

# mse = mean_squared_error(y_test, y_predict)
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test,y_predict))
# rmse = RMSE(y_test, y_predict)
# print(f"RMSE : {rmse : .2f}")


