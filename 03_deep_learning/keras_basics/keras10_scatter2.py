import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.data
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=129
)

#2.model
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(7))
model.add(Dense(1))

#3.compile, train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=385, batch_size=3)
print("=====================================")

#4.evaluate,predict
loss = model.evaluate(x_test,y_test)
print('loss = ', loss)

results = model.predict(x)

plt.scatter(x,y)
plt.plot(x, results, color='green')
plt.show()
