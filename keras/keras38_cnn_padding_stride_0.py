import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten


#2. model
model = Sequential()
model.add(Conv2D(10,(2,2), input_shape=(10,10,1), #(None, 10, 10, 10)
        strides=2, padding='same',  #stride=1이 디폴트값
                 ))
model.add(Conv2D(filters=9, kernel_size=(3,3), #(None, 8, 8, 9)  
         strides=3,padding='valid')) #padding valid가 디폴트값
model.summary()












