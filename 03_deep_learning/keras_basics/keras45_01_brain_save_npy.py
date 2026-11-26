
import numpy as np
import time 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout
from tensorflow.python.keras.layers import Flatten,GlobalAveragePooling2D
from sklearn.metrics import accuracy_score #이진분류
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.callbacks import EarlyStopping
#1.data

train_datagen = ImageDataGenerator(
    rescale=1./255, #형변환
    # horizontal_flip = True, #수평 뒤집기,
    # vertical_flip = True, #수직 뒤집기
    # width_shift_range = 0.1, #평행이동
    # height_shift_range = 0.1, 
    # rotation_range = 5, #각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.2,
    # shear_range = 0.7, #좌표하나하나 고정하고 다른 몇개 좌표를 이동
    # fill_mode = 'nearest',

) #클래스

test_datagen = ImageDataGenerator(
    rescale=1./255,
) #평가 훈련할 데이터는 rescale만 하면됨. 시험지는 변환시키면 안됨 항상.

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, #경로
    target_size=(150,150), #마음대로 써도 알아서 늘리거나 줄여줌 , 모든 사진들이 크기가 다르기때문에 알아서 맞춤
    batch_size=160, #이미지 batchsize () -> 통배치할때는 그냥 큰숫자 넣으면 됨 .1000넣어도 통배치임
    class_mode='binary', #이진분류
    color_mode='grayscale', #흑백
    shuffle=True,
)

#Found 160 images belonging to 2 classes.//

xy_test = test_datagen.flow_from_directory(
    path_test, #경로
    target_size=(150,150), #마음대로 써도 알아서 늘리거나 줄여줌 , 모든 사진들이 크기가 다르기때문에 알아서 맞춤
    batch_size=120, #이미지 batchsize ()
    class_mode='binary', #이진분류
    color_mode='grayscale', #흑백
    shuffle=False, #필요가 없음 test는 섞을 필요가 없음
)

# Found 120 images belonging to 2 classes
# print(xy_train[0][0].shape) #(160, 200, 200, 1)
# print(xy_train[0][1].shape) #(160,)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

# print(x_train.shape, y_train.shape) #(160, 150, 150, 1) (160,)
# print(x_test.shape, y_test.shape) #(120, 150, 150, 1) (120,)

np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path + 'keras45_01_x_train.npy', arr=x_train)
np.save(np_path + 'keras45_01_y_train.npy', arr=y_train)
np.save(np_path + 'keras45_01_x_test.npy', arr=x_test)
np.save(np_path + 'keras45_01_y_test.npy', arr=y_test)

exit()
#2.model

model = Sequential()

model.add(Conv2D(16, (3,3), input_shape=(150, 150, 1),activation='relu', padding='same', strides=1))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), activation='relu', padding='same', strides=1))
model.add(Conv2D(32, (3,3), activation='relu', padding='same', strides=1))
model.add(Conv2D(16, (3,3), activation='relu', padding='same', strides=1))
model.add(Conv2D(16, (3,3), activation='relu', padding='same', strides=1))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

#3.compile,train
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',          # validation loss 감시
    mode='auto',                  # val_loss는 작을수록 좋음
    patience=300,                 # 10 epoch 동안 개선 없으면 중단
    restore_best_weights=True,   # 가장 좋았던 weight로 복구
    verbose=1
)
start_time = time.time()

model.fit(x_train ,y_train, epochs=1000, batch_size=16,
          verbose=1,callbacks=[es],
          validation_split=0.2,
          )

end_time = time.time()

#4.evaluate, predict
print('===============model.evaluate==============')
loss = model.evaluate(x_test, y_test, verbose=1,)
print('loss : ', loss)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')













