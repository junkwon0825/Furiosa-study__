import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Conv1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping


#1.data

ROOT = Path(r"C:\study")
DATA_PATH = ROOT / "_data" / "samsung"
SAVE_PATH = ROOT / "_save" / "keras70"

SAVE_PATH.mkdir(parents=True, exist_ok=True)
#keras70 저장 폴더 없으면 만들어. 이미 있으면 그냥 넘어가.

SAMSUNG_PATH = DATA_PATH / "삼성전자261012.csv"
AMORE_PATH = DATA_PATH / "아모레261012.csv"
HYNIX_PATH = DATA_PATH / "하이닉스261012.csv"

np.random.seed(42)
tf.random.set_seed(42)


def read_csv(path):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    return df

#CSV를 모델에 넣을 수 있도록 청소해주는 함수
def preprocessing(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns] #모든 컬럼 이름을 문자열로 만들고 양옆 공백을 제거.
    columns = list(df.columns)

    if "대비" in columns:
        idx = columns.index("대비")

        if idx + 1 < len(columns):
            next_col = columns[idx + 1]

            if next_col.startswith("Unnamed") or next_col == "" or next_col.startswith("대비."):
                df = df.rename(columns={next_col: "대비금액"})

        df = df.drop(columns=["대비"])

    df["일자"] = pd.to_datetime(df["일자"], errors="coerce")

    for col in df.columns:
        if col == "일자":
            continue

        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(",", "", regex=False)
        df[col] = df[col].str.replace("%", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(axis=1, how="all")
    df = df.dropna()
    df = df.drop_duplicates(subset=["일자"])
    df = df.sort_values("일자").reset_index(drop=True)

    return df


samsung = preprocessing(read_csv(SAMSUNG_PATH))
amore = preprocessing(read_csv(AMORE_PATH))
hynix = preprocessing(read_csv(HYNIX_PATH))


# 세 종목에 모두 존재하는 날짜만 사용
common_dates = set(samsung["일자"]) & set(amore["일자"]) & set(hynix["일자"])

samsung = samsung[samsung["일자"].isin(common_dates)].sort_values("일자").reset_index(drop=True)
amore = amore[amore["일자"].isin(common_dates)].sort_values("일자").reset_index(drop=True)
hynix = hynix[hynix["일자"].isin(common_dates)].sort_values("일자").reset_index(drop=True)

print("공통 날짜 개수 :", len(common_dates))
print("삼성전자 :", samsung.shape)
print("아모레 :", amore.shape)
print("하이닉스 :", hynix.shape)


# 컬럼 앞에 종목명 붙이기
samsung = samsung.rename(columns={col: f"삼성전자_{col}" for col in samsung.columns if col != "일자"})
amore = amore.rename(columns={col: f"아모레_{col}" for col in amore.columns if col != "일자"})
hynix = hynix.rename(columns={col: f"하이닉스_{col}" for col in hynix.columns if col != "일자"})


# 날짜 기준 통합
dataset = pd.merge(samsung, amore, on="일자", how="inner")
dataset = pd.merge(dataset, hynix, on="일자", how="inner")
dataset = dataset.sort_values("일자").reset_index(drop=True)

dataset.to_csv(DATA_PATH / "삼성_아모레_하이닉스_통합.csv", index=False, encoding="utf-8-sig")

print("통합 데이터 :", dataset.shape)
print("첫 날짜 :", dataset["일자"].iloc[0])
print("마지막 날짜 :", dataset["일자"].iloc[-1])


# x = 삼성 + 아모레 + 하이닉스 전체 변수
feature_columns = [col for col in dataset.columns if col != "일자"]
x = dataset[feature_columns].values.astype(np.float32)

# y 계산에 사용할 삼성전자 시가 / 종가
samsung_open = dataset["삼성전자_시가"].values.astype(np.float32)
samsung_close = dataset["삼성전자_종가"].values.astype(np.float32)

n_features = x.shape[1]
timesteps = 20

x_data = []
y_data = []
base_close_data = []

for i in range(len(x) - timesteps):
    x_data.append(x[i:i + timesteps])

    today_close = samsung_close[i + timesteps - 1]
    tomorrow_open = samsung_open[i + timesteps]

    y_data.append(tomorrow_open / today_close - 1)
    base_close_data.append(today_close)

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32).reshape(-1, 1)
base_close_data = np.array(base_close_data, dtype=np.float32).reshape(-1, 1)

print("x_data :", x_data.shape)
print("y_data :", y_data.shape)

print("전체 y 변화율 최소 :", y_data.min())
print("전체 y 변화율 최대 :", y_data.max())


# 시간 순서 유지해서 split
x_train, x_test, y_train, y_test, close_train, close_test = train_test_split(
    x_data, y_data, base_close_data, test_size=0.2, shuffle=False
)

print("x_train :", x_train.shape)
print("x_test :", x_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)


# Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_train = scaler_x.fit_transform(x_train.reshape(-1, n_features)).reshape(x_train.shape)
x_test = scaler_x.transform(x_test.reshape(-1, n_features)).reshape(x_test.shape)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)


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


#3.compile,train

model_lstm.compile(loss="mse", optimizer="adam")
model_gru.compile(loss="mse", optimizer="adam")
model_conv.compile(loss="mse", optimizer="adam")

es1 = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
es2 = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
es3 = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)

history1 = model_lstm.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[es1], shuffle=False)
history2 = model_gru.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[es2], shuffle=False)
history3 = model_conv.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[es3], shuffle=False)


#4.evaluate,predict

# 각 모델이 예측한 것은 "다음날 시가 변화율"
pred_lstm = model_lstm.predict(x_test)
pred_gru = model_gru.predict(x_test)
pred_conv = model_conv.predict(x_test)

# Scaling 복원
pred_lstm = scaler_y.inverse_transform(pred_lstm)
pred_gru = scaler_y.inverse_transform(pred_gru)
pred_conv = scaler_y.inverse_transform(pred_conv)
y_test_return = scaler_y.inverse_transform(y_test)


# 변화율 → 실제 삼성전자 시가로 변환
y_test_price = close_test * (1 + y_test_return)

pred_lstm_price = close_test * (1 + pred_lstm)
pred_gru_price = close_test * (1 + pred_gru)
pred_conv_price = close_test * (1 + pred_conv)


# Ensemble 변화율
pred_ensemble = (pred_lstm + pred_gru + pred_conv) / 3

# Ensemble 변화율 → 삼성전자 실제 가격
pred_ensemble_price = close_test * (1 + pred_ensemble)


# 실제 가격 기준 RMSE
rmse_lstm = np.sqrt(mean_squared_error(y_test_price, pred_lstm_price))
rmse_gru = np.sqrt(mean_squared_error(y_test_price, pred_gru_price))
rmse_conv = np.sqrt(mean_squared_error(y_test_price, pred_conv_price))
rmse_ensemble = np.sqrt(mean_squared_error(y_test_price, pred_ensemble_price))

print("\n========================================")
print("모델 성능 - 삼성전자 시가 RMSE")
print("========================================")

print("LSTM RMSE     :", rmse_lstm)
print("GRU RMSE      :", rmse_gru)
print("Conv1D RMSE   :", rmse_conv)
print("Ensemble RMSE :", rmse_ensemble)


# 모델 weight 저장
model_lstm.save_weights(SAVE_PATH / "lstm.weights.h5")
model_gru.save_weights(SAVE_PATH / "gru.weights.h5")
model_conv.save_weights(SAVE_PATH / "conv1d.weights.h5")

print("\n모델 weight 저장 완료")


# 테스트 마지막 10개 확인
print("\n실제 삼성 시가")
print(y_test_price[-10:].reshape(-1))

print("\nLSTM")
print(pred_lstm_price[-10:].reshape(-1))

print("\nGRU")
print(pred_gru_price[-10:].reshape(-1))

print("\nConv1D")
print(pred_conv_price[-10:].reshape(-1))

print("\nEnsemble")
print(pred_ensemble_price[-10:].reshape(-1))