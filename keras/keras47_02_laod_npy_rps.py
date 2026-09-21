
import numpy as np
import time 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout
from tensorflow.python.keras.layers import Flatten,GlobalAveragePooling2D
from sklearn.metrics import accuracy_score #이진분류
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
#1.data
path = './_data/image/rps/'
save_path = './_save/keras47/'
np_path = './_data/kaggle_cat_dog_npy/'


x_train = np.load(np_path + 'keras46_02_x_train.npy')
y_train = np.load(np_path + 'keras46_02_y_train.npy')
x_test = np.load(np_path + 'keras46_02_x_test.npy')
y_test = np.load(np_path + 'keras46_02_y_test.npy')

#2.model

model = Sequential()

model.add(Conv2D(16, (5,5), input_shape=(150, 150, 3),activation='relu', padding='same', strides=1))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (4,4), activation='relu', padding='same', strides=1))
model.add(Dropout(0.2))
model.add(Conv2D(32, (4,4), activation='relu', padding='same', strides=1))
model.add(Dropout(0.2))
model.add(Conv2D(16, (4,4), activation='relu', padding='same', strides=1))
model.add(Conv2D(16, (4,4), activation='relu', padding='same', strides=1))

model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D())

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3.compile,train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
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

model.fit(x_train ,y_train, epochs=1000, batch_size=16,
          verbose=1,callbacks=[es],
          validation_split=0.2,
          )

end_time = time.time()
model.save_weights(save_path + 'final.weights.h5')


#4.evaluate, predict
print('===============model.evaluate==============')
loss = model.evaluate(x_test, y_test, verbose=1,)
print('loss : ', loss)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')











