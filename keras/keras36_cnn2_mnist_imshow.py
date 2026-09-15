import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt


(x_train,y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,) #실제데이터는 뒤에 1이 생략. 4차원데이터임
print(x_test.shape,y_test.shape) #(10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

print(pd.value_counts(y_test))

plt.imshow(x_train[680], 'gray')
plt.show()



