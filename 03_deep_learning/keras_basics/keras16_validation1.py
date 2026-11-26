#15 copy
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#7:3으로 나눌거임.
x_train = np.array([1,2,3,4,5,6])
y_train = np.array([1,2,3,4,5,6])

x_val = np.array([7,8])
y_val = np.array([7,8])


x_test = np.array([9,10])
y_test = np.array([9,10])

#순수하게 그냥 손으로 나눈방법. 앞으로 보게될 변수명.train,test데이터

#2.model
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


#3.compile,train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=4, 
          verbose=2,  
          validation_data = (x_val, y_val),
          )

# 훈련 부분 -> verbose
# verbose= 0 : 침묵
# verbose = 1 : 디폴트
# verbose = 2 : 프로그래스바 삭제
# verbose >= 3 : epoch만 나옴

#4.evaluate,predict
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

#훈련을 해보지 않은 로스 . 좀더 신뢰가 가는 로스 x,y보다
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 53.4349 학습하고 있는 로스
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step - loss: 230.0000 훈련안한 데이터 로스
# x_test,y_test loss 를 신뢰해야함.