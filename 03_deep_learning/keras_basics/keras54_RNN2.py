#54-1 copy
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau


#1.data
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1) 
print(x.shape) #(7,3,1)

#2.model
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3,1)))
model.add(SimpleRNN(30, input_shape=(9,1)))
#3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, ))
model.add(Dense(50, ))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.summary()

#파라미터의 개수 = units*feature + units*bias + units*units//
#timesteps 값은 전혀 영향 미치지 않는다.
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 30)                960       
                                                                 
#  dense (Dense)               (None, 80)                2480      
                                                                 
#  dense_1 (Dense)             (None, 100)               8100      
                                                                 
#  dense_2 (Dense)             (None, 80)                8080      
                                                                 
#  dense_3 (Dense)             (None, 50)                4050      
                                                                 
#  dense_4 (Dense)             (None, 30)                1530      
                                                                 
#  dense_5 (Dense)             (None, 20)                620       
                                                                 
#  dense_6 (Dense)             (None, 1)                 21        
                                                                 
# =================================================================
# Total params: 25,841
# Trainable params: 25,841
# Non-trainable params: 0
# _________________________________________________________________