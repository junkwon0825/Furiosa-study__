# Day 27 - Ensemble Stock Prediction

## 오늘 핵심

오늘은 **삼성전자, 아모레퍼시픽, SK하이닉스의 여러 주가 데이터를 사용해서 다음 거래일 삼성전자 시가를 예측하는 Ensemble 모델**을 만들었다.

최종 구조는 다음과 같다.

```text
삼성전자 + 아모레 + 하이닉스 최근 20거래일 데이터
                    ↓
            전처리 / Scaling
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
      LSTM         GRU        Conv1D
        ↓           ↓           ↓
    변화율 예측   변화율 예측   변화율 예측
        └───────────┼───────────┘
                    ↓
                 평균
                    ↓
          Ensemble 변화율
                    ↓
       마지막 삼성전자 종가 × (1 + 변화율)
                    ↓
        다음 거래일 삼성전자 예상 시가
```

---

## 1. Ensemble이란?

Ensemble은 여러 모델의 예측 결과를 합쳐 최종 예측값을 만드는 방법이다.

이번 프로젝트에서는 세 모델을 각각 따로 학습했다.

```python
model_lstm.fit(x_train, y_train, ...)
model_gru.fit(x_train, y_train, ...)
model_conv.fit(x_train, y_train, ...)
```

세 모델은 같은 `x_train`, `y_train`을 사용하지만 내부 weight는 서로 다르게 학습된다.

최종 예측은 단순 평균을 사용했다.

```python
pred_ensemble = (pred_lstm + pred_gru + pred_conv) / 3
```

현재 프로젝트는 **Weighted Ensemble이 아니라 동일 비율 평균 Ensemble**이다.

---

## 2. Multi-output과 현재 모델의 차이

Multi-output은 하나의 모델에서 여러 정답을 출력하는 구조다.

```text
공통 입력
   ↓
공통 Layer
   ↓
 ┌─────┴─────┐
 ↓           ↓
y1          y2
```

Keras에서는:

```python
model = Model(inputs=input1, outputs=[output1, output2])
```

현재 삼성전자 예측 프로젝트는 Multi-output이 아니라:

```text
LSTM   → 삼성 시가 변화율 1개
GRU    → 삼성 시가 변화율 1개
Conv1D → 삼성 시가 변화율 1개
```

처럼 **Single-output 모델 3개를 결합한 Ensemble 구조**다.

---

## 3. 경로 설정

```python
ROOT = Path(r"C:\study")
DATA_PATH = ROOT / "_data" / "samsung"
SAVE_PATH = ROOT / "_save" / "keras70"
```

`Path`는 Windows 파일 경로를 편하게 만들기 위해 사용한다.

```python
SAVE_PATH.mkdir(parents=True, exist_ok=True)
```

뜻:

```text
keras70 폴더가 없으면 생성
이미 있으면 오류 없이 넘어감
```

---

## 4. CSV 읽기

```python
def read_csv(path):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    return df
```

먼저 UTF-8로 읽고, 한글 인코딩 오류가 발생하면 CP949로 다시 읽는다.

---

## 5. 데이터 전처리

### 컬럼 이름 공백 제거

```python
df.columns = [str(col).strip() for col in df.columns]
```

예:

```text
" 시가 " → "시가"
```

### 대비 컬럼 처리

원본 CSV에서 `대비` 컬럼에 ▲/▼가 들어 있고 바로 오른쪽 컬럼에 실제 금액 차이가 들어 있었다.

따라서:

```python
df = df.rename(columns={next_col: "대비금액"})
df = df.drop(columns=["대비"])
```

로 처리했다.

결과:

```text
▲ / ▼ 컬럼 → 제거
-7000, +5000 같은 실제 값 → 대비금액으로 사용
```

### 날짜 변환

```python
df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
```

문자열 날짜를 실제 날짜형 데이터로 변환한다.

### 숫자 변환

```python
df[col] = df[col].astype(str).str.strip()
df[col] = df[col].str.replace(",", "", regex=False)
df[col] = df[col].str.replace("%", "", regex=False)
df[col] = pd.to_numeric(df[col], errors="coerce")
```

예:

```text
"284,500" → 284500
"2.31%" → 2.31
```

### 결측값 / 중복 제거

```python
df = df.dropna(axis=1, how="all")
df = df.dropna()
df = df.drop_duplicates(subset=["일자"])
```

### 날짜 정렬

```python
df = df.sort_values("일자").reset_index(drop=True)
```

주식은 시계열 데이터이므로 반드시 과거 → 최신 순서로 정렬한다.

---

## 6. 세 종목의 날짜 맞추기

세 CSV의 행 개수는 서로 다를 수 있다.

단순히 가장 짧은 데이터 길이에 맞춰 자르면 날짜가 어긋날 수 있다.

그래서:

```python
common_dates = set(samsung["일자"]) & set(amore["일자"]) & set(hynix["일자"])
```

로 세 종목에 모두 존재하는 날짜만 구한다.

그 후:

```python
samsung = samsung[samsung["일자"].isin(common_dates)]
```

처럼 공통 거래일만 남긴다.

핵심:

```text
행 개수를 강제로 맞추는 것이 아니라
같은 날짜를 기준으로 shape을 맞춘다.
```

---

## 7. 종목명을 컬럼에 붙이기

원래는 세 CSV 모두:

```text
시가
고가
저가
종가
거래량
...
```

처럼 같은 컬럼명을 가지고 있다.

그래서:

```python
삼성전자_시가
삼성전자_종가
아모레_시가
아모레_종가
하이닉스_시가
하이닉스_종가
```

형태로 이름을 변경한다.

```python
samsung = samsung.rename(columns={col: f"삼성전자_{col}" for col in samsung.columns if col != "일자"})
```

---

## 8. 세 종목 병합

```python
dataset = pd.merge(samsung, amore, on="일자", how="inner")
dataset = pd.merge(dataset, hynix, on="일자", how="inner")
```

`how="inner"`는 세 데이터에서 동일하게 존재하는 날짜만 병합한다는 의미다.

통합 CSV도 저장했다.

```python
dataset.to_csv(DATA_PATH / "삼성_아모레_하이닉스_통합.csv", index=False, encoding="utf-8-sig")
```

---

## 9. X 데이터 구성

```python
feature_columns = [col for col in dataset.columns if col != "일자"]
x = dataset[feature_columns].values.astype(np.float32)
```

`일자`를 제외한 삼성전자 + 아모레 + 하이닉스의 모든 숫자 데이터를 입력 X로 사용한다.

예를 들어 feature가 총 45개라면:

```text
x.shape = (전체 거래일 수, 45)
```

---

## 10. 왜 y를 삼성전자 시가 자체가 아니라 변화율로 바꿨는가?

처음에는:

```python
y = dataset["삼성전자_시가"]
```

처럼 다음날 시가 자체를 정답으로 사용했다.

하지만 확인 결과:

```text
Train 삼성 시가 : 약 50,200 ~ 111,800원
Test 삼성 시가  : 약 95,600 ~ 372,500원
```

처럼 Train과 Test의 가격 범위 차이가 너무 컸다.

그래서 모델이 과거 가격 수준에 머물며 최근 높은 가격을 잘 예측하지 못했다.

이를 개선하기 위해 최종 모델에서는:

```text
다음날 시가 자체
```

가 아니라:

```text
오늘 종가 대비 다음날 시가 변화율
```

을 y로 사용했다.

---

## 11. 변화율 y 만들기

```python
today_close = samsung_close[i + timesteps - 1]
tomorrow_open = samsung_open[i + timesteps]

y_data.append(tomorrow_open / today_close - 1)
```

예:

```text
오늘 종가 = 280,000원
다음날 시가 = 284,500원
```

계산:

```text
284500 / 280000 - 1
= 0.01607
= +1.607%
```

따라서 모델은 `284,500원`이라는 절대 가격 대신 `+1.607%`라는 변화율을 학습한다.

---

## 12. timestep = 20

```python
timesteps = 20
```

모델은 최근 20거래일을 보고 다음 거래일을 예측한다.

```python
x_data.append(x[i:i + timesteps])
```

예:

```text
1일 ~ 20일 → 21일 예측
2일 ~ 21일 → 22일 예측
3일 ~ 22일 → 23일 예측
```

따라서 LSTM/GRU/Conv1D 입력 shape은:

```text
(N, 20, n_features)
```

이다.

---

## 13. base_close_data

```python
base_close_data.append(today_close)
```

각 데이터의 기준 종가를 저장한다.

모델이 예측하는 값은 변화율이므로 실제 가격으로 복원하려면:

```text
오늘 종가 × (1 + 예측 변화율)
```

이 필요하다.

---

## 14. Train / Test Split

```python
x_train, x_test, y_train, y_test, close_train, close_test = train_test_split(
    x_data, y_data, base_close_data, test_size=0.2, shuffle=False
)
```

X, y, 기준 종가를 한 번에 split해서 같은 날짜끼리 정렬이 유지되도록 한다.

`shuffle=False`를 사용하는 이유:

```text
과거 → Train
최근 → Test
```

라는 시계열 순서를 유지하기 위해서다.

---

## 15. Scaling

```python
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
```

### X Scaling

```python
x_train = scaler_x.fit_transform(x_train.reshape(-1, n_features)).reshape(x_train.shape)
x_test = scaler_x.transform(x_test.reshape(-1, n_features)).reshape(x_test.shape)
```

LSTM 데이터는 3차원:

```text
(N, 20, features)
```

이지만 `MinMaxScaler`는 2차원 입력을 받기 때문에:

```text
3D → 2D → Scaling → 3D
```

순서로 처리한다.

### fit / transform / inverse_transform

```text
fit
= 데이터의 최소/최대 기준을 학습

transform
= 그 기준으로 데이터를 변환

fit_transform
= fit + transform

inverse_transform
= Scaling 값을 원래 단위로 복원
```

Test 데이터는 Train 기준으로만 변환한다.

---

## 16. LSTM

```python
input1 = Input(shape=(timesteps, n_features))
x1 = LSTM(64)(input1)
x1 = Dense(32, activation="relu")(x1)
x1 = Dropout(0.2)(x1)
output1 = Dense(1)(x1)
```

LSTM은 시간 순서가 있는 데이터에서 과거 정보를 기억하면서 패턴을 학습한다.

출력 `Dense(1)`은 다음날 삼성전자 시가 변화율 하나를 출력한다.

---

## 17. GRU

```python
x2 = GRU(64)(input2)
```

GRU도 시계열 모델이며 LSTM과 비슷하지만 내부 gate 구조가 더 단순하다.

이번 Ensemble에서는 LSTM과 다른 방식으로 시계열 특징을 학습하도록 사용한다.

---

## 18. Conv1D

```python
x3 = Conv1D(128, 3, padding="same", activation="relu")(input3)
x3 = Conv1D(64, 3, padding="same", activation="relu")(x3)
x3 = Conv1D(32, 3, padding="same", activation="relu")(x3)
```

Conv1D는 시간축에서 가까운 구간의 패턴을 찾는다.

`kernel_size=3`:

```text
3개의 timestep씩 보면서 특징을 추출
```

`padding="same"`:

```text
입력 timestep 길이를 유지
```

한다.

---

## 19. EarlyStopping

```python
EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
```

뜻:

```text
val_loss를 관찰
↓
50 epoch 동안 좋아지지 않으면 학습 종료
↓
가장 좋았던 weight로 복원
```

`restore_best_weights=True`가 있기 때문에 마지막 epoch의 weight가 아니라 가장 성능이 좋았던 weight를 사용한다.

---

## 20. 세 모델은 각각 따로 학습

```python
history1 = model_lstm.fit(...)
history2 = model_gru.fit(...)
history3 = model_conv.fit(...)
```

즉:

```text
같은 x_train, y_train
         ↓
 ┌───────┼───────┐
 ↓       ↓       ↓
LSTM    GRU    Conv1D
각자 독립적으로 학습
```

한다.

---

## 21. Scaling 복원

모델 prediction은 Scaling된 y 값이다.

```python
pred_lstm = scaler_y.inverse_transform(pred_lstm)
```

을 이용해 실제 변화율로 복원한다.

예:

```text
모델 출력 0.63
↓ inverse_transform
실제 변화율 +0.012
↓
+1.2%
```

---

## 22. 변화율 → 실제 삼성전자 시가

```python
pred_lstm_price = close_test * (1 + pred_lstm)
```

예:

```text
오늘 종가 = 280,000원
예측 변화율 = +1.2%
```

이면:

```text
280000 × 1.012
= 283,360원
```

이 다음 거래일 예상 시가가 된다.

---

## 23. Ensemble

```python
pred_ensemble = (pred_lstm + pred_gru + pred_conv) / 3
```

세 모델의 변화율 예측을 동일한 비중으로 평균한다.

예:

```text
LSTM   : +1.0%
GRU    : +1.3%
Conv1D : +0.8%
```

이면 Ensemble:

```text
(+1.0 + 1.3 + 0.8) / 3
= 약 +1.03%
```

---

## 24. RMSE

```python
rmse_lstm = np.sqrt(mean_squared_error(y_test_price, pred_lstm_price))
```

RMSE는 실제 삼성전자 시가와 예측 시가의 차이를 측정한다.

```text
RMSE가 낮을수록 더 좋은 모델
```

이다.

---

## 25. Weight 저장

```python
model_lstm.save_weights(SAVE_PATH / "lstm.weights.h5")
model_gru.save_weights(SAVE_PATH / "gru.weights.h5")
model_conv.save_weights(SAVE_PATH / "conv1d.weights.h5")
```

`save_weights()`는 모델 전체가 아니라 학습된 weight만 저장한다.

따라서 Predict 코드에서는 똑같은 모델 구조를 다시 만든 다음:

```python
model_lstm.load_weights(...)
```

로 weight를 넣어야 한다.

---

## 26. Predict 전용 코드

Predict 파일에는:

```python
model.fit()
```

이 없다.

즉 모델을 다시 학습하지 않는다.

```text
통합 CSV 불러오기
↓
SAVE 당시와 같은 Scaling 기준 생성
↓
최신 20거래일 준비
↓
같은 모델 구조 생성
↓
가중치 3개 load
↓
각 모델 predict
↓
변화율 inverse_transform
↓
세 변화율 평균
↓
마지막 삼성 종가 × (1 + Ensemble 변화율)
↓
삼성전자 다음 거래일 예상 시가
```

최종 출력:

```python
print(f'삼성전자 예측 가격 "{final_price:,.0f}원"')
```
