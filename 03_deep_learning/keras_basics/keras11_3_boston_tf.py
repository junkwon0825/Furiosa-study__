from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

#1.data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)

#2.model
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(30))
model.add(Dense(62))
model.add(Dense(48))
model.add(Dense(23))
model.add(Dense(1))

#3.compile, train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=375, batch_size = 34)

print("==========================================")
#4.evaluate, predict
loss = model.evaluate(x_train,y_train)
print('loss : ', loss)

results = model.predict(x_test)

#loss :  26.41722869873047








