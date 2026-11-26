
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
path = './_data/image/men_women/'
save_path = './_save/keras47/'
np_path = './_data/kaggle_cat_dog_npy/'

x_train = np.load(np_path + 'keras47_01_x_train.npy')
y_train = np.load(np_path + 'keras47_01_y_train.npy')
x_test = np.load(np_path + 'keras47_01_x_test.npy')
y_test = np.load(np_path + 'keras47_01_y_test.npy')

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




