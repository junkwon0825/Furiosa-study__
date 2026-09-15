#36-2copy
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt

#1.data
(x_train,y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,) #실제데이터는 뒤에 1이 생략. 4차원데이터임
print(x_test.shape,y_test.shape) #(10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) #255 0
print(np.max(x_test), np.min(x_test)) #255 0

##### 스케일링 1
x_train = x_train/255. #.붙이면 float형태로 출력
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0
print(np.max(x_test), np.min(x_test))

##### 스케일링 2 ->이미지 데이터에서 많이 사용 (-1~1사이로)
x_train = (x_train-127.5)/127.5 #.붙이면 float형태로 출력
x_test = (x_test-127.5)/127.5
print(np.max(x_train), np.min(x_train)) #-1.0 1.0
print(np.max(x_test), np.min(x_test)) #-1.0 1.0



#2.model

#3.compile,train

