#https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data
import pandas as pd
import numpy as np
import time 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.layers import Flatten,GlobalAveragePooling2D,BatchNormalization
from sklearn.metrics import accuracy_score #이진분류
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.callbacks import EarlyStopping

#1. data
path = "./_data/kaggle_cat_dog/"

path_train = './_data/kaggle_cat_dog/train/'
path_test = './_data/kaggle_cat_dog/test/'
submission = pd.read_csv(path + "sample_submission.csv", index_col=0)

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

xy_train = train_datagen.flow_from_directory(
    path_train, #경로
    target_size=(100,100), #마음대로 써도 알아서 늘리거나 줄여줌 , 모든 사진들이 크기가 다르기때문에 알아서 맞춤
    batch_size=8100, #이미지 batchsize () -> 통배치할때는 그냥 큰숫자 넣으면 됨 .1000넣어도 통배치임
    class_mode='binary', #이진분류
    color_mode='rgb', #흑백
    shuffle=True,
)

# Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test, #경로
    target_size=(100,100), #마음대로 써도 알아서 늘리거나 줄여줌 , 모든 사진들이 크기가 다르기때문에 알아서 맞춤
    batch_size=1200, #이미지 batchsize ()
    class_mode='binary', #이진분류
    color_mode='rgb', #흑백
    shuffle=False, #필요가 없음 test는 섞을 필요가 없음
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape) 