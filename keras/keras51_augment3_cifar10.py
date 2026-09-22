#36-2copy
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
#1.data
(x_train,y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.
x_test = x_test / 255.

################################### 데이터 증폭 ###########################
datagen = ImageDataGenerator(
    # rescale=1./255, #형변환
    # horizontal_flip = True, #수평 뒤집기, (좌우반전)
    # vertical_flip = True, #수직 뒤집기 (상하 반전)
    # width_shift_range = 0.1, #평행이동
    height_shift_range = 0.1, 
    rotation_range = 15, #각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.1,
    # shear_range = 0.7, #좌표하나하나 고정하고 다른 몇개 좌표를 이동
    fill_mode = 'nearest',

) #클래스 전체 적용하면 사진 이상해짐

augment_size = 40000 
print(x_train.shape[0]) #60000 임 

randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False) # 6만개 중에 4만개 랜덤뽑가
print(randidx) #[44278   641 58142 ... 24781  5774 19062] 벡터형태임 
# print(randidx.shape) #(40000,) 벡터니까 먹혔다.
# print(len(randidx)) #40000 리스트는 len으로 확인해야하는데 벡터도 먹힌다.

print(np.min(randidx), np.max(randidx)) #0 59997 그냥 랜던값 ㅋㅋ
x_augmented = x_train[randidx].copy() #메모리변수 안전한 공간 생성
y_augmented = y_train[randidx].copy() #x,y똑같은 위치 잡아줌

# print(x_augmented.shape, y_augmented.shape) #(40000, 28, 28) (40000,)
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],3)  #(40000, 28, 28, 1)
# print(x_augmented.shape) #(40000, 28, 28, 1)000

xy_augmented = datagen.flow(
            x_augmented, y_augmented, 
            batch_size=augment_size,
            shuffle=False,
).next()[0]

#### 4만장 변환 완료 ####
# print(x_augmented.shape) #(40000, 28, 28, 1)

# print(x_train.shape) # (60000,28,28)
# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)

#### 데이터 합치기 ###
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# print(x_train.shape,y_train.shape)  #(100000, 28, 28, 1) (100000,)

# print(np.unique(y_train, return_counts=True))
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

#2.model
model = Sequential()

model.add(Conv2D(128, (3,3), input_shape=(32,32,3),activation='relu', padding='same'))
model.add(Conv2D(128, (3,3),activation='relu', padding='same',))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), activation='relu', padding='same',))
model.add(Conv2D(64, (3,3), activation='relu', padding='same', strides=2))
model.add(Dropout(0.25))

model.add(Conv2D(32, (2,2), activation='relu', padding='same',))

model.add(Conv2D(30, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(GlobalAveragePooling2D())


model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3.compile,train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',          # validation loss 감시
    mode='auto',                  # val_loss는 작을수록 좋음
    patience=50,                 # 10 epoch 동안 개선 없으면 중단
    restore_best_weights=True,   # 가장 좋았던 weight로 복구
    verbose=1
)
start_time = time.time()
model.fit(x_train ,y_train, epochs=100, batch_size=200,
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
y_predict = np.argmax(y_predict, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')



# accuracy_score :  0.9894
# 걸린시간 :  319.98 초