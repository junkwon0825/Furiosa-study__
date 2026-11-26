from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
history = model.fit(x_train,y_train, epochs=345, batch_size = 30,
                    verbose=1, validation_split=0.33)

print("==========================================")
#4.evaluate, predict
loss = model.evaluate(x_train,y_train)
print('loss : ', loss)

results = model.predict(x_test)

print("====================== history ===============================")
plt.figure(figsize=(9,6)) #그냥 그림판 자체 크기 사이즈
plt.plot(history.history['loss'][30:], c='red', label='loss') #y값만 넣으면 x디폴트는 시간순으로 그려줌
plt.plot(history.history['val_loss'][30:], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('boston Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid() 
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False
plt.show()
#loss :  26.41722869873047
