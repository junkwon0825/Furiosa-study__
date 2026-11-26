from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#2. model
model = Sequential()
model.add(Dense(800, input_dim=200, activation='relu'))
model.add(Dense(600, activation='relu')) 
model.add(Dense(400)) 
model.add(Dense(200, activation='relu'))
model.add(Dense(100))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax')) 

model.summary()




