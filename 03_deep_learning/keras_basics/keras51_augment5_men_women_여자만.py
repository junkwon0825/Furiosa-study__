
import numpy as np
import time 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout
from tensorflow.python.keras.layers import Flatten,GlobalAveragePooling2D
from sklearn.metrics import accuracy_score #이진분류
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
#1.data
# path = './_data/image/men_women/'
# save_path = './_save/keras47/'
# np_path = './_data/kaggle_cat_dog_npy/'

# x_train = np.load(np_path + 'keras47_01_x_train.npy')
# y_train = np.load(np_path + 'keras47_01_y_train.npy')
# x_test = np.load(np_path + 'keras47_01_x_test.npy')
# y_test = np.load(np_path + 'keras47_01_y_test.npy')

path = './_data/image/men_women/faces/'
save_path = './_save/keras47/'
np_path = './_data/kaggle_cat_dog_npy/'
datagen = ImageDataGenerator(
    rescale=1./255
)
datagen_aug = ImageDataGenerator(
    # rescale=1./255, #형변환
    # horizontal_flip = True, #수평 뒤집기, (좌우반전)
    # vertical_flip = True, #수직 뒤집기 (상하 반전)
    width_shift_range = 0.1, #평행이동
    height_shift_range = 0.1, 
    rotation_range = 15, #각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.1,
    # shear_range = 0.7, #좌표하나하나 고정하고 다른 몇개 좌표를 이동
    fill_mode = 'nearest',

) 

xy_data = datagen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=3000,      # 전체 이미지보다 크게
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

# 같은 batch에서 x, y를 같이 꺼냄
# Found 27167 images belonging to 2 classes.
x, y = xy_data[0]

print(x.shape) #(2000, 100, 100, 3)
print(y.shape) #(2000,)

# train / test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y)

#numpy데이터 분류
# x_train_w = x_train[np.where(y_train > 0.0)]
# y_train_w = y_train[np.where(y_train > 0.0)]

################################### 여자 데이터 증폭 ###########################
women_idx = (y_train == 1)

x_women = x_train[women_idx]
y_women = y_train[women_idx]

print(x_women.shape, y_women.shape) #(856, 100, 100, 3) (856,)

augment_size = 800
print(x_train.shape[0]) #1600

randidx = np.random.choice(x_women.shape[0], size=augment_size, replace=False) # 6만개 중에 4만개 랜덤뽑가
# print(randidx) #[44278   641 58142 ... 24781  5774 19062] 벡터형태임 
# print(randidx.shape) #(40000,) 벡터니까 먹혔다.
print(len(randidx)) #300 리스트는 len으로 확인해야하는데 벡터도 먹힌다.

print(np.min(randidx), np.max(randidx)) #0 59997 그냥 랜던값 ㅋㅋ
x_w_augmented = x_women[randidx].copy()
y_w_augmented = y_women[randidx].copy()

print(x_w_augmented.shape, y_w_augmented.shape) #(300, 100, 100, 3) (300,)

x_augmented = x_w_augmented.reshape(x_w_augmented.shape[0],
                                  x_w_augmented.shape[1],
                                  x_w_augmented.shape[2],3)  
# print(x_augmented.shape) #(40000, 32, 32, 3)

x_augmented,y_augmented = datagen_aug.flow(
            x_w_augmented, y_w_augmented, 
            batch_size=augment_size,
            shuffle=False,
).next()

#### 4만장 변환 완료 ####
# print(x_augmented.shape) #(40000, 32, 32, 3)

# print(x_train.shape) # (50000, 32, 32, 3)
# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)

#### 데이터 합치기 ###
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape) #(1600, 150, 150, 3) (1600,)
print(x_test.shape, y_test.shape) #(400, 150, 150, 3) (400,)

np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path + 'keras47_01_x_train.npy', arr=x_train)
np.save(np_path + 'keras47_01_y_train.npy', arr=y_train)
np.save(np_path + 'keras47_01_x_test.npy', arr=x_test)
np.save(np_path + 'keras47_01_y_test.npy', arr=y_test)

#2.model

model = Sequential()

model.add(Conv2D(32, (5,5), input_shape=(100, 100, 3),activation='relu', padding='same', strides=1))
model.add(Conv2D(64, (5,5), activation='relu', padding='same', strides=1))
model.add(Conv2D(64, (5,5), activation='relu', padding='same', strides=1))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (5,5), activation='relu', padding='same', strides=1))
model.add(Conv2D(32, (5,5), activation='relu', padding='same', strides=1))
model.add(Conv2D(16, (5,5), activation='relu', padding='same', strides=1))

model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D())

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# model.load_weights(save_path + 'final.weights.h5') #실제 save가중치

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
np_path = './_data/kaggle_cat_dog_npy/'
arr = np.load(np_path + "keras48_oo.npy")
arr = arr/255.
print('===============model.evaluate==============')
loss = model.evaluate(x_test, y_test, verbose=1,)
print('loss : ', loss)

y_predict = model.predict(arr)[0][0]

print("예측값 :", y_predict)

if y_predict < 0.5:
    print("남자")
else:
    print("여자")




