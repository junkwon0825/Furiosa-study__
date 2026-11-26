import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Conv1D, Flatten, Dropout


#1.data

ROOT = Path(r"C:\study")
DATA_PATH = ROOT / "_data" / "samsung"
SAVE_PATH = ROOT / "_save" / "keras70"

dataset = pd.read_csv(DATA_PATH / "삼성_아모레_하이닉스_통합.csv", encoding="utf-8-sig")

x = dataset.drop(columns=["일자"]).values.astype(np.float32)

samsung_open = dataset["삼성전자_시가"].values.astype(np.float32)
samsung_close = dataset["삼성전자_종가"].values.astype(np.float32)

timesteps = 20
n_features = x.shape[1]

x_data = []
y_data = []

for i in range(len(x) - timesteps):
    x_data.append(x[i:i + timesteps])

    today_close = samsung_close[i + timesteps - 1]
    tomorrow_open = samsung_open[i + timesteps]

    y_data.append(tomorrow_open / today_close - 1)

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32).reshape(-1, 1)


# SAVE 때와 동일한 train 범위
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=False)


# SAVE 때와 동일한 scaler 기준 재현
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(x_train.reshape(-1, n_features))
scaler_y.fit(y_train)


# 가장 최신 20거래일
x_pred = x[-timesteps:]
x_pred = scaler_x.transform(x_pred)
x_pred = x_pred.reshape(1, timesteps, n_features)

# 다음 거래일 시가 계산의 기준이 되는 오늘 종가
last_close = samsung_close[-1]

print("마지막 삼성전자 종가 :", last_close)
print("예측 입력 shape :", x_pred.shape)


#2.model

# LSTM
input1 = Input(shape=(timesteps, n_features))
x1 = LSTM(64)(input1)
x1 = Dense(32, activation="relu")(x1)
x1 = Dropout(0.2)(x1)
output1 = Dense(1)(x1)
model_lstm = Model(inputs=input1, outputs=output1)


# GRU
input2 = Input(shape=(timesteps, n_features))
x2 = GRU(64)(input2)
x2 = Dense(32, activation="relu")(x2)
x2 = Dropout(0.2)(x2)
output2 = Dense(1)(x2)
model_gru = Model(inputs=input2, outputs=output2)


# Conv1D
input3 = Input(shape=(timesteps, n_features))
x3 = Conv1D(128, 3, padding="same", activation="relu")(input3)
x3 = Conv1D(64, 3, padding="same", activation="relu")(x3)
x3 = Conv1D(32, 3, padding="same", activation="relu")(x3)
x3 = Flatten()(x3)
x3 = Dense(64, activation="relu")(x3)
x3 = Dense(32, activation="relu")(x3)
x3 = Dropout(0.2)(x3)
output3 = Dense(1)(x3)
model_conv = Model(inputs=input3, outputs=output3)


# 저장한 weight 불러오기
model_lstm.load_weights(SAVE_PATH / "lstm.weights.h5")
model_gru.load_weights(SAVE_PATH / "gru.weights.h5")
model_conv.load_weights(SAVE_PATH / "conv1d.weights.h5")


#3.compile,train
# Predict 전용이므로 compile / fit 없음


#4.evaluate,predict

# 모델이 예측하는 것은 다음날 시가 변화율
pred_lstm = model_lstm.predict(x_pred)
pred_gru = model_gru.predict(x_pred)
pred_conv = model_conv.predict(x_pred)

pred_lstm = scaler_y.inverse_transform(pred_lstm)[0][0]
pred_gru = scaler_y.inverse_transform(pred_gru)[0][0]
pred_conv = scaler_y.inverse_transform(pred_conv)[0][0]


# Weighted Ensemble
pred_ensemble = (pred_lstm + pred_gru + pred_conv) / 3

# 변화율 → 실제 삼성전자 다음 거래일 시가
price_lstm = last_close * (1 + pred_lstm)
price_gru = last_close * (1 + pred_gru)
price_conv = last_close * (1 + pred_conv)

final_price = last_close * (1 + pred_ensemble)


print("\n========================================")
print("삼성전자 다음 거래일 시가 예측")
print("========================================")

print(f"LSTM 변화율 : {pred_lstm * 100:.3f}%")
print(f"GRU 변화율 : {pred_gru * 100:.3f}%")
print(f"Conv1D 변화율 : {pred_conv * 100:.3f}%")

print()

print(f"LSTM 예상 시가 : {price_lstm:,.0f}원")
print(f"GRU 예상 시가 : {price_gru:,.0f}원")
print(f"Conv1D 예상 시가 : {price_conv:,.0f}원")

print("----------------------------------------")
print(f'삼성전자 예측 가격 "{final_price:,.0f}원"')