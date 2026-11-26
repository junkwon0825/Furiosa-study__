# 69-3 copy
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1.data
x1_datasets = np.array([range(100), range(301, 401)]).T #(100,2)
                       # 삼성 종가         하이닉스 종가
                                
y1 = np.array(range(3001, 3101))
                # 화성의 화씨 온도
y2 = np.array(range(13001, 13101))


x1_train, x1_test,\
y1_train, y1_test,y2_train, y2_test = train_test_split(
    x1_datasets,
    y1,
    y2,
    train_size=0.8,
    shuffle=True,
    random_state=273
)

#2-1.model
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name = 'han1')(input1)
dense2 = Dense(20, activation='relu', name = 'han2')(dense1)
dense3 = Dense(30, activation='relu', name = 'han3')(dense2)
dense4 = Dense(40, activation='relu', name = 'han4')(dense3)
output1 = Dense(5, activation='relu', name = 'han5')(dense4)
# model1 = Model(inputs = input1, outputs = output1)


# 분기1 : y1
last_dense1 = Dense(20, activation='relu', name='ld1')(dense4)
last_dense2 = Dense(10, activation='relu', name='ld2')(last_dense1)
last_output1 = Dense(1, name='last1')(last_dense2)

# 분기2 : y2
last_dense3 = Dense(20, activation='relu', name='ld3')(dense4)
last_dense4 = Dense(10, activation='relu', name='ld4')(last_dense3)
last_output2 = Dense(1, name='last2')(last_dense4)

model = Model(inputs=input1, outputs=[last_output1,last_output2]) #앙상블 모델 구현 완
# model.summary()


#3.compile,train
model.compile(loss='mse', optimizer='adam')

model.fit(
    x1_train,
    [y1_train, y2_train],
    epochs=100,
    batch_size=8
)

#4.평가 예측
result = model.evaluate(x1_test,[y1_test, y2_test])
print('loss : ', result)

x1_pred = np.array([range(100,106), range(400,406)]).T

y_pred1, y_pred2 = model.predict(x1_pred)

print('예측값 :', y_pred1,y_pred2)
