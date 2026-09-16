import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score

#1.data
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape,y_train.shape) 
print(x_test.shape,y_test.shape) 

print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))

#스케일링
x_train = (x_train-127.5)/127.5
x_test = (x_test-127.5)/127.5
# print(np.max(x_train), np.min(x_train)) #1.0 -1.0
# print(np.max(x_test), np.min(x_test)) #1.0 -1.0

# x값 4차원으로 변환
x_train = x_train.reshape(-1,32,32,3)
x_test = x_test.reshape(-1,32,32,3)
# print(x_train.shape, x_test.shape,) #(10000, 28, 28, 1) (60000, 28, 28, 1)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = y_test.reshape(-1,1)
y_test = ohe.fit_transform(y_test)

# print(y_test.shape,y_train.shape) #(10000, 10) (60000, 10)

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i+1)

    # -1~1 → 0~1로 되돌려서 출력
    plt.imshow((x_train[i] + 1) / 2)

    # one-hot → 클래스 번호
    label = np.argmax(y_train[i])

    plt.title(class_names[label])
    plt.axis('off')

plt.tight_layout()
plt.show()

#2.model
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(32,32,3)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(20, (2,2), activation='relu'))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.summary()


#3.compile,train
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

start_time = time.time()
model.fit(x_train ,y_train, epochs=100, batch_size=120,
          verbose=1,
          validation_split=0.2,
          )

end_time = time.time()

#4.evaluate, predict
print('===============model.evaluate==============')
loss = model.evaluate(x_test, y_test, verbose=1,)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)
y_pred_class = np.argmax(y_predict, axis=1)
y_true = y_test.reshape(-1)

wrong = np.where(y_pred_class != y_true)[0]
print(wrong[:10])

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')

idx = wrong[0]

plt.imshow(x_test[idx])

plt.title(
    f"True: {class_names[y_true[idx]]} / "
    f"Pred: {class_names[y_pred_class[idx]]}"
)

plt.show()