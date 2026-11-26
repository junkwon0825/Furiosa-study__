from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

#1.data
x = np.array(range(1,17))
y = np.array(range(1,17))


x_train, x_test, y_train , y_test = train_test_split(x, y, train_size=0.75,
                                                     shuffle=True,
                                                     random_state=34,
                                                     )


#2. model
model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(7))
model.add(Dense(1))

#3.compile , train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=380, batch_size=4, verbose=1, validation_split=0.33)

#4.evaluate,predict
loss = model.evaluate(x_test,y_test)
print("loss : " , loss)

