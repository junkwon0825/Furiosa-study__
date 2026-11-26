import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])


# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

x_train = x[:7]
y_train = y[:7]

x_test = x[7:]
y_test = y[7:]

#2.model
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


#3.compile,train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=380, batch_size=4)

#4.evaluate,predict
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

#훈련을 해보지 않은 로스 . 좀더 신뢰가 가는 로스 x,y보다
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 53.4349 학습하고 있는 로스
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step - loss: 230.0000 훈련안한 데이터 로스
# x_test,y_test loss 를 신뢰해야함.