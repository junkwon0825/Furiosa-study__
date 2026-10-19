import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

print(tf.__version__)

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1.data
x = np.array([1,2,3,4,5,])
y = np.array([1,2,3,4,5,])

#2.model
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
print("=================== model.weights ========================")
print(model.weights)
print("=================== model.trainable weights ========================")
print(model.trainable_weights)
print("===========================================")
print(len(model.weights)) # 6
print(len(model.trainable_weights))  # 6

######################### 동결 #######################
model.trainable = False  # 동결 
print("===========================================")
print(len(model.weights)) # 6
print(len(model.trainable_weights))  # 0
print("=================== model.weights ========================")
print(model.weights)
print("=================== model.trainable weights ========================")
print(model.trainable_weights)

model.summary()