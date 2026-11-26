from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

#1.data
x = np.array(range(1,17))
y = np.array(range(1,17))

# 실습 8개 ,4개 ,4개 잘라보자
x_train = x[:8]
y_train = y[:8]

x_val = x[8:12]
y_val = y[8:12]

x_test = x[12:]
y_test = y[12:]

print(x_train.shape, x_val.shape, x_test.shape)

# x_train,x_val, x_test, y_train,y_val, y_test = train_test_split(x, y,
#                                                      train_size=0.7,
#                                                      test_size=0.3,
#                                                      shuffle=True, #디폴트 섞는다.
#                                                      random_state=34,
#                                                      )



#2. model
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

