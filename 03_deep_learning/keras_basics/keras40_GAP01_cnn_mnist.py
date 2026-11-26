#36-2copy
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
#1.data
(x_train,y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,) #실제데이터는 뒤에 1이 생략. 4차원데이터임
# print(x_test.shape,y_test.shape) #(10000, 28, 28) (10000,)

# print(np.max(x_train), np.min(x_train)) #255 0
# print(np.max(x_test), np.min(x_test)) #255 0

# ##### 스케일링 1
# x_train = x_train/255. #.붙이면 float형태로 출력
# x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) #1.0 0.0
# print(np.max(x_test), np.min(x_test))

##### 스케일링 2 ->이미지 데이터에서 많이 사용 (-1~1사이로)
x_train = (x_train-127.5)/127.5 #.붙이면 float형태로 출력
x_test = (x_test-127.5)/127.5
# print(np.max(x_train), np.min(x_train)) #1.0 -1.0
# print(np.max(x_test), np.min(x_test)) #1.0 -1.0

# print(y_test.shape,y_train.shape) #(10000,) (60000,)

# x값 4차원으로 변환
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# print(x_test.shape,x_train.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(60000,1)
y_train = y_train.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = y_test.reshape(-1,1)
y_test = ohe.fit_transform(y_test)

# print(y_test.shape,y_train.shape) #(10000, 10) (60000, 10)


#2.model
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(28,28,1), activation='relu', padding='same')) #(26,26,64)
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(60,(3,3),activation='relu', padding='same')) #(22,22,16)
model.add(Dropout(0.5))
model.add(Conv2D(10,(2,2),activation='relu')) #(21,21,16)
model.add(Dropout(0.4))
model.add(Conv2D(30,(2,2),activation='relu', padding='same')) #(20,20,16)
model.add(Conv2D(30,(2,2),activation='relu')) #(21,21,16)

model.add(Flatten()) #2차원변경
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax')) # (10,)
model.summary()

#3.compile,train
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',          # validation loss 감시
    mode='auto',                  # val_loss는 작을수록 좋음
    patience=20,                 # 10 epoch 동안 개선 없으면 중단
    restore_best_weights=True,   # 가장 좋았던 weight로 복구
    verbose=1
)
start_time = time.time()
model.fit(x_train ,y_train, epochs=100, batch_size=120,
          verbose=1,callbacks=[es],
          validation_split=0.2,
          )


end_time = time.time()

#4.evaluate, predict
print('===============model.evaluate==============')
loss = model.evaluate(x_test, y_test, verbose=1,)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')



# accuracy_score :  0.9894
# 걸린시간 :  319.98 초