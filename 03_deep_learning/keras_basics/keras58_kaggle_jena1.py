import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ,ReduceLROnPlateau

# 1. DATA
path = "./_data/kaggle_jena/"

dataset = pd.read_csv(
    path + 'jena_climate_2009_2016.csv',
    index_col=0
)

print(dataset.shape)
# (420551, 14)
print(dataset.head())
print(dataset.columns)

# 마지막 하루 144개의 실제 풍향
# 마지막에 예측값과 비교할 정답
y_cor = dataset[-144:]['wd (deg)']

# x는 하루 전 데이터
# 마지막 컬럼인 풍향은 제거
x_data = dataset[:-288].drop( ['wd (deg)'],axis=1)

# y는 x보다 144칸 뒤의 풍향
y_data = dataset[144:-144]['wd (deg)']

print('x_data :', x_data.shape)
print('y_data :', y_data.shape)
print('y_cor :', y_cor.shape)

# x_data : (420263, 13)
# y_data : (420263,)
# y_cor  : (144,)

# 마지막 하루 바로 전날의 144개 데이터
x_predict = dataset[-288:-144].drop(['wd (deg)'],axis=1)

print('x_predict :', x_predict.shape)
# (144, 13)

x_data = np.array(x_data)
y_data = np.array(y_data)

x_predict = np.array(x_predict)
y_cor = np.array(y_cor)

# 420263개는 144로 나누어 떨어지지 않음
# 앞의 71개를 제거하면 144개 단위로 맞음
x_data = x_data[71:]
y_data = y_data[71:]

print(x_data.shape)
print(y_data.shape)


x = x_data.reshape(-1, 144, 13)
y = y_data.reshape(-1, 144, 1)

x_predict = x_predict.reshape(1, 144, 13)

y_cor = y_cor.reshape(144, 1)

print('x shape :', x.shape)
print('y shape :', y.shape)

print('x_predict shape :', x_predict.shape)
print('y_cor shape :', y_cor.shape)

# x         : (2918, 144, 13)
# y         : (2918, 144, 1)
# x_predict : (1, 144, 13)
# y_cor     : (144, 1)

# 2. MODEL
model = Sequential()
model.add(LSTM(100, input_shape=(144,13),return_sequences=True))
model.add(LSTM(300,return_sequences=True))
model.add(LSTM(500,return_sequences=True))
model.add(LSTM(500,return_sequences=True))
model.add(Dense(300, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))


#3.compile, train

learning_rate = 0.001

model.compile(
    loss='mse',
    optimizer=Adam(
        learning_rate=learning_rate
    ),
    metrics=['mae']
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=1000,
    factor=0.5,
    verbose=1
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=1000,
    restore_best_weights=True,
    verbose=1
)

model.fit(
    x,
    y,
    epochs=1000,
    batch_size=400,
    validation_split=0.2,
    callbacks=[rlr, es],
    shuffle=False
)


#4. PREDICT, EVALUATE
results = model.evaluate(x, y)

print('loss : ', results)

y_predict = model.predict(x_predict)

print('예측 shape :', y_predict.shape)
y_predict = y_predict.reshape(144, 1)


print('============================')
print('예측 풍향')
print(y_predict)

print('============================')
print('실제 풍향')
print(y_cor)
