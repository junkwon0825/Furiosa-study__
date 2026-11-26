import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1.data
x1_datasets = np.array([range(100), range(301, 401)]).T #(100,2)
                       # 삼성 종가         하이닉스 종가
x2_datasets = np.array([range(101,201), range(411,511), # (100,3)
                        # 원유가            환율
                        range(150,250)]).transpose()
                           # 금시세
                                
y = np.array(range(3001, 3101))
                # 화성의 화씨 온도

x1_train, x1_test, x2_train, x2_test,\
y_train, y_test = train_test_split(
    x1_datasets,
    x2_datasets,
    y,
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

#2-2.model
input11 = Input(shape=(3,))
dense11 = Dense(50, name = 'han11')(input11)
dense12 = Dense(40, name = 'han12')(dense11)
dense13 = Dense(30, name = 'han13')(dense12)
dense14 = Dense(20, name = 'han14')(dense13)
output11 = Dense(3, name = 'han15')(dense14)
# model2 = Model(inputs = input11, outputs = output11)

#2-3 model합치기
from tensorflow.keras.layers import concatenate, Concatenate

# merge1 = concatenate([output1, output11], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output11])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1,input11], outputs=last_output) #앙상블 모델 구현 완
# model.summary()


#3.compile,train
model.compile(loss='mse', optimizer='adam')

model.fit(
    [x1_train, x2_train],
    y_train,
    epochs=100,
    batch_size=8
)

#4.평가 예측
result = model.evaluate([x1_test, x2_test],y_test)
print('loss : ', result)

x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516),
                    range(249,255)]).T

y_pred = model.predict([x1_pred, x2_pred])

print('예측값 :', y_pred)
