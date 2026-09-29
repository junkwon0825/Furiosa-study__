# Day20 - RNN 다중 예측, 시계열 분할, Jena Climate, RAG 기초

## 1. 오늘 학습 내용

오늘은 RNN 계열 모델에서 **미래값 1개 예측**을 넘어 **여러 개의 미래값을 예측하는 방법**을 학습했다.

- 100까지의 데이터를 이용해 101 하나가 아니라 101~106까지 예측
- `split_x`, `split_xy`를 이용한 시계열 데이터 분할
- GRU/LSTM을 이용한 다중 출력
- `return_sequences=True`의 의미
- DNN / CNN / RNN 차이
- Jena Climate 풍향 예측
- GPU OOM과 `batch_size`
- RAG 기본 아키텍처

---

## 2. 여러 개의 미래값 한 번에 예측

기존 방식:

```text
91 ~ 100
   ↓
 Model
   ↓
  101
```

다중 출력 방식:

```text
91 ~ 100
   ↓
 Model
   ↓
101 ~ 106
```

예측하고 싶은 값이 6개라면 마지막 출력층을 다음처럼 만든다.

```python
model.add(Dense(6))
```

즉,

```text
Dense(1) → 1개 예측
Dense(6) → 6개 예측
```

### 코드

```python
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GRU

#1.data
a = np.array(range(1,101))
a = a.reshape(50,2)

size = 8

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i+size]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)

x = bbb[:, :5, :]
y = bbb[:, 5:, :]
y = y.reshape(y.shape[0], 6)

print(x.shape)   # (43,5,2)
print(y.shape)   # (43,6)


#2.model
model = Sequential()

model.add(GRU(30, input_shape=(5,2)))
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20, activation='relu'))
model.add(Dense(6))


#3.compile,train
learning_rate = 0.008

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
model.fit(x, y, epochs=1000)


#4.evaluate,predict
results = model.evaluate(x,y)
print('loss : ', results)

x_predict = np.array(range(91,101)).reshape(1,5,2)
y_predict = model.predict(x_predict)

print('101~106 예측 결과 : ', y_predict)
```

---

## 3. split_xy를 이용한 다음 값 예측

```python
def split_xy(dataset, size):
    x = []
    y = []

    for i in range(len(dataset) - size):
        x_subset = dataset[i:i+size]
        y_subset = dataset[i+size]

        x.append(x_subset)
        y.append(y_subset)

    return np.array(x), np.array(y)
```

예:

```text
[1,2,3,4,5,6,7,8,9,10] → 11
[2,3,4,5,6,7,8,9,10,11] → 12
[3,4,5,6,7,8,9,10,11,12] → 13
```

### 코드

```python
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GRU

#1.data
a = np.array(range(1,101))

size = 10

def split_xy(dataset, size):
    x = []
    y = []

    for i in range(len(dataset) - size):
        x_subset = dataset[i:i+size]
        y_subset = dataset[i+size]

        x.append(x_subset)
        y.append(y_subset)

    return np.array(x), np.array(y)

x, y = split_xy(a, size)

print(x.shape)   # (90,10)
print(y.shape)   # (90,)

x = x.reshape(x.shape[0], x.shape[1], 1)
x = x.reshape(x.shape[0], 5, 2)

print(x.shape)   # (90,5,2)


#2.model
model = Sequential()

model.add(GRU(30, input_shape=(5,2)))
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))


#3.compile,train
learning_rate = 0.008

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
model.fit(x, y, epochs=1000)


#4.evaluate,predict
results = model.evaluate(x, y)
print('loss : ', results)

x_predict = np.array(range(91,106))

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i+size]
        aaa.append(subset)
    return np.array(aaa)

x_predict = split_x(x_predict, 10)

print(x_predict)
print(x_predict.shape)   # (6,10)

x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
x_predict = x_predict.reshape(x_predict.shape[0], 5, 2)

print(x_predict.shape)   # (6,5,2)

y_predict = model.predict(x_predict)

print('101~106 예측 결과 : ')
print(y_predict)
```

> 주의: 위 두 번째 예시는 102~106 예측에 실제 101~105 값이 입력에 들어간다.  
> 실제 미래값이 없는 상황에서 이전 예측값을 다시 입력으로 쓰는 재귀적 예측과는 다르다.

---

## 4. DNN / CNN / RNN 비교

| 구분 | DNN | CNN | RNN |
|---|---|---|---|
| 주 용도 | 일반 수치/표 데이터 | 이미지 | 시계열/순서 데이터 |
| 대표 입력 shape | `(N, features)` | `(N, H, W, C)` | `(N, T, F)` |
| 대표 Layer | `Dense` | `Conv2D` | `SimpleRNN`, `LSTM`, `GRU` |
| 입력 특징 | feature | 가로×세로×채널 | 시간×feature |
| 순서 중요성 | 보통 낮음 | 공간 위치 중요 | 시간/순서 매우 중요 |
| 회귀 출력 예 | `Dense(1)` | `Dense(1)` | `Dense(1)` |
| 10클래스 분류 | `Dense(10)` | `Dense(10)` | `Dense(10)` |

핵심:

```text
DNN → feature
CNN → 공간
RNN → 시간 / 순서
```

---

## 5. RNN을 여러 층으로 쌓기

중간 RNN 층이 다음 RNN 층에 모든 timestep 정보를 넘기려면:

```python
return_sequences=True
```

를 사용한다.

```python
model = Sequential()

model.add(LSTM(10, input_shape=(3,1), return_sequences=True))
model.add(LSTM(20, return_sequences=True))
model.add(LSTM(15, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(5, return_sequences=True))
model.add(LSTM(5))
model.add(Dense(1))
```

보통:

```text
중간 RNN → return_sequences=True
마지막 RNN → return_sequences=False
```

하지만 **모든 timestep마다 출력을 만들어야 하는 문제**라면 마지막 RNN도 `True`가 필요할 수 있다.

---

## 6. LSTM 예제

```python
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1.data
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0], x.shape[1], 1)


#2.model
model = Sequential()

model.add(LSTM(16, input_shape=(3,1)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


#3.compile,train
learning_rate = 0.001

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto',
                        patience=200, verbose=1, factor=0.5)

es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience=500, restore_best_weights=True, verbose=1)

model.fit(x, y, epochs=1000, batch_size=4,
          validation_split=0.2, callbacks=[rlr, es])


#4.evaluate,predict
results = model.evaluate(x,y)
print('loss : ', results)

x_predict = np.array([50,60,70]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[50,60,70] 다음 예측의 결과 : ', y_predict)
```

회귀 문제에서는 `accuracy`보다 `mse`, `mae`가 더 적절하다.

---

## 7. Jena Climate 풍향 예측

Jena Climate 데이터는 10분 간격이다.

```text
1시간 = 6개
1일 = 24 × 6 = 144개
```

목표:

```text
하루의 기상 데이터 144개
        ↓
      LSTM
        ↓
다음 하루의 풍향 144개 예측
```

### 데이터 구성

```python
#1.data
path = "./_data/kaggle_jena/"

dataset = pd.read_csv(
    path + 'jena_climate_2009_2016.csv',
    index_col=0
)

print(dataset.shape)
# (420551,14)

y_cor = dataset[-144:]['wd (deg)']

x_data = dataset[:-288].drop(['wd (deg)'], axis=1)
y_data = dataset[144:-144]['wd (deg)']

x_predict = dataset[-288:-144].drop(['wd (deg)'], axis=1)

x_data = np.array(x_data)
y_data = np.array(y_data)
x_predict = np.array(x_predict)
y_cor = np.array(y_cor)

x_data = x_data[71:]
y_data = y_data[71:]

x = x_data.reshape(-1,144,13)
y = y_data.reshape(-1,144,1)

x_predict = x_predict.reshape(1,144,13)
y_cor = y_cor.reshape(144,1)

print('x shape :', x.shape)
print('y shape :', y.shape)
print('x_predict shape :', x_predict.shape)
print('y_cor shape :', y_cor.shape)
```

shape:

```text
x         : (2918,144,13)
y         : (2918,144,1)
x_predict : (1,144,13)
y_cor     : (144,1)
```

---

## 8. Jena 모델에서 중요한 shape

현재 모델:

```python
#2.model
model = Sequential()

model.add(LSTM(100, input_shape=(144,13), return_sequences=True))
model.add(LSTM(300, return_sequences=True))
model.add(LSTM(500, return_sequences=True))
model.add(LSTM(500))

model.add(Dense(300, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
```

마지막 LSTM이:

```python
LSTM(500)
```

이면 `return_sequences=False`가 기본값이므로 최종 recurrent 출력은:

```text
(N,500)
```

이 되고 마지막 Dense 이후:

```text
(N,1)
```

이 된다.

만약 144개의 풍향을 모두 출력하려면 마지막 recurrent layer에서도 timestep을 유지해야 한다.

```python
model.add(LSTM(500, return_sequences=True))
model.add(Dense(1))
```

그러면 출력은:

```text
(N,144,1)
```

이 된다.

---

## 9. Jena 학습

```python
#3.compile,train
learning_rate = 0.001

model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=learning_rate),
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
    epochs=2000,
    batch_size=400,
    validation_split=0.2,
    callbacks=[rlr, es],
    shuffle=False
)
```

---

## 10. Evaluate / Predict

```python
#4.evaluate,predict
results = model.evaluate(x,y)
print('loss : ', results)

y_predict = model.predict(x_predict)

print('예측 shape : ', y_predict.shape)
print('예측 풍향')
print(y_predict)

print('실제 풍향')
print(y_cor)
```

---

## 11. GPU OOM

대표 오류:

```text
OOM when allocating tensor
```

OOM:

```text
Out Of Memory
```

즉 GPU 메모리 부족이다.

### batch_size와 GPU 메모리

```text
batch_size ↑
→ 한 번에 처리하는 데이터 ↑
→ GPU 메모리 사용량 ↑

batch_size ↓
→ 한 번에 처리하는 데이터 ↓
→ GPU 메모리 사용량 ↓
```

OOM이 나면:

```python
batch_size=32
```

그래도 안 되면:

```python
batch_size=16
```

또는:

```python
batch_size=8
```

### GPU 메모리 점진적 사용

TensorFlow import 전에:

```python
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
```

를 넣을 수 있다.

단, 이 설정을 사용해도 모델이나 batch가 너무 크면 OOM은 발생할 수 있다.

---

## 12. RAG 기본 아키텍처

RAG:

```text
Retrieval-Augmented Generation
```

LLM이 바로 답변하지 않고 외부 문서를 검색한 뒤 그 내용을 참고해서 답변하는 구조이다.

```text
문서
 ↓
Chunking
 ↓
Embedding
 ↓
Vector DB
```

질문이 들어오면:

```text
사용자 질문
    ↓
Embedding
    ↓
Vector DB 검색
    ↓
관련 문서
    ↓
질문 + 관련 문서
    ↓
LLM
    ↓
답변
```

일반 LLM:

```text
질문 → LLM → 답변
```

RAG:

```text
질문 → 검색 → 관련 문서 → LLM → 답변
```

핵심:

> RAG는 LLM이 답변하기 전에 외부 지식을 검색해서 참고하도록 만드는 구조이다.

---

## 13. 오늘 핵심 정리

```text
Dense(1)
→ 1개 예측

Dense(6)
→ 6개 예측
```

```text
RNN 입력 shape
→ (N, T, F)
```

```text
중간 RNN
→ return_sequences=True
```

```text
모든 timestep 출력 필요
→ 마지막 RNN도 return_sequences=True
```

```text
split_x
→ 일정 길이 window 생성

split_xy
→ x window + 다음 값 y 생성
```

```text
Jena
144개 = 하루
오늘 기상정보 → 다음날 풍향
```

```text
OOM
→ batch_size 줄이기
```

```text
RAG
질문 → 검색 → 관련 문서 → LLM → 답변
```
