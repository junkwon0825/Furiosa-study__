
import numpy as np
import time 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout
from tensorflow.python.keras.layers import Flatten,GlobalAveragePooling2D
from sklearn.metrics import accuracy_score #이진분류
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint

#1.data
datagen = ImageDataGenerator(
    rescale=1./255
)
path = './_data/image/men_women/faces/'
save_path = './_save/keras47/'
np_path = './_data/kaggle_cat_dog_npy/'
xy_data = datagen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=2000,      # 전체 이미지보다 크게
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

# 같은 batch에서 x, y를 같이 꺼냄
# Found 27167 images belonging to 2 classes.
x, y = xy_data[0]

print(x.shape)
print(y.shape)

# train / test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y)


print(x_train.shape, y_train.shape) #(1600, 150, 150, 3) (1600,)
print(x_test.shape, y_test.shape) #(400, 150, 150, 3) (400,)

np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path + 'keras47_01_x_train.npy', arr=x_train)
np.save(np_path + 'keras47_01_y_train.npy', arr=y_train)
np.save(np_path + 'keras47_01_x_test.npy', arr=x_test)
np.save(np_path + 'keras47_01_y_test.npy', arr=y_test)

#2.model

model = Sequential()

model.add(Conv2D(16, (5,5), input_shape=(150, 150, 3),activation='relu', padding='same', strides=1))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (5,5), activation='relu', padding='same', strides=1))
model.add(Conv2D(32, (5,5), activation='relu', padding='same', strides=1))
model.add(Conv2D(16, (5,5), activation='relu', padding='same', strides=1))

model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D())

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
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=True,
    filepath=save_path + 'best.weights.h5',
    verbose=1
)
start_time = time.time()

model.fit(x_train ,y_train, epochs=1000, batch_size=800,
          verbose=1,callbacks=[es],
          validation_split=0.2,
          )

end_time = time.time()
model.save_weights(save_path + 'final.weights.h5')


#4.evaluate, predict
print('===============model.evaluate==============')
loss = model.evaluate(x_test, y_test, verbose=1,)
print('loss : ', loss)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')












