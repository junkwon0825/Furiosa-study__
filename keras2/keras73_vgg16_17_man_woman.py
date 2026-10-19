import numpy as np
import time
import tensorflow as tf
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)


#1.data

path = './_data/image/men_women/faces/'
save_path = './_save/keras47/'

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

xy_train = datagen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=64,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    subset='training',
    seed=333
)

xy_val = datagen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=64,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,
    subset='validation',
    seed=333
)

print(xy_train.class_indices)
print(xy_train.samples)
print(xy_val.samples)


#2.model

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(100,100,3)
)

vgg16.trainable = False

model = Sequential()

model.add(vgg16)

model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()


#3.compile,train

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=True,
    filepath=save_path + 'vgg16_best.weights.h5',
    verbose=1
)

start_time = time.time()

history = model.fit(
    xy_train,
    epochs=300,
    verbose=1,
    callbacks=[es, mcp],
    validation_data=xy_val
)

end_time = time.time()

model.save_weights(save_path + 'vgg16_final.weights.h5')


#4.evaluate,predict

print('===============model.evaluate==============')

loss = model.evaluate(xy_val, verbose=1)

print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(xy_val)

y_predict = (y_predict > 0.5).astype(int).reshape(-1)

y_test = xy_val.classes

from sklearn.metrics import accuracy_score

acc_score = accuracy_score(y_test, y_predict)

print('accuracy_score :', acc_score)
print('걸린시간 :', round(end_time - start_time, 2), '초')