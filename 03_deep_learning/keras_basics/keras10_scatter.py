import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                     train_size=0.7,
                                                     test_size=0.3,
                                                     shuffle=True, #디폴트 섞는다.
                                                     random_state=294,
                                                     )

print('x_train :', x_train)
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)

#2. model
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(7))
model.add(Dense(1))

#3.compile , train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=380, batch_size=4)

print("=====================================") #evaluate 최종 w 값에만 집어넣어서 결과 판단 구분선
#4.evaluate,predict
loss = model.evaluate(x_test,y_test)
print("loss : " , loss)

results = model.predict(x)

# 데이터 시각화.그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x,y) #데이터 점찍기
plt.plot(x,results, color='rainbow')
plt.show()
