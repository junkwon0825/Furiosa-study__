#50-1copy
#data 증폭시켜서 성능향상 목표

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score


(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()
################################### 데이터 증폭 ###########################
datagen = ImageDataGenerator(
    rescale=1./255, #형변환
    # horizontal_flip = True, #수평 뒤집기, (좌우반전)
    # vertical_flip = True, #수직 뒤집기 (상하 반전)
    # width_shift_range = 0.1, #평행이동
    height_shift_range = 0.1, 
    rotation_range = 15, #각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.1,
    # shear_range = 0.7, #좌표하나하나 고정하고 다른 몇개 좌표를 이동
    fill_mode = 'nearest',

) #클래스 전체 적용하면 사진 이상해짐

augment_size = 100 #100개 증가시키겠다

print(x_train.shape) #(60000,28,28)
print(x_train[0].shape) #(28,28)

# aaa = np.tile(x_train[0], augment_size)
# print(aaa.shape) #(28, 2800)


aaa = np.tile(x_train[0], augment_size).reshape(-1,28,28,1)
print(aaa.shape) #(100, 28, 28, 1)
### 단순 복붙 ### 이대로 쓰면 모델 망함
## 다른 수치로 작업 imagedatagenerator 사용##
## 여기 위에 있는 데이터로 작업하고 싶으면 flow. directory 데이터 꺼내고 싶으면 flow_from directory
xy_data = datagen.flow(
        np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),
        np.zeros(augment_size), #다 0채워넣으셈 100개데이터 y값에
        batch_size=augment_size,
        shuffle=False,
).next()

print(xy_data)
print(type(xy_data)) #<class 'tuple'>
# print(xy_data.shape) #AttributeError: 'tuple' object has no attribute 'shape' 튜플은 쉐잎이 없음
print(len(xy_data)) #2 이유는 x,y 2개이기 때문
print(xy_data[0].shape) #(100, 28, 28, 1)
print(xy_data[1].shape) #(100,)

plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(xy_data[0][i], cmap='gray')
plt.show()









