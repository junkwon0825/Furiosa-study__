# Day19 — RNN, LSTM, GRU, 시계열 데이터 분할

## 1. 오늘 학습 흐름

오늘은 시계열 데이터를 다루는 RNN 계열 모델과, RNN 학습을 위해 데이터를 일정 길이로 자르는 방법을 배웠다.

```text
SimpleRNN
↓
Long Sequence 문제
↓
Gradient Vanishing / Exploding
↓
LSTM
↓
GRU
↓
시계열 데이터 Sliding Window
↓
RNN 입력용 x, y 구성
```

---

## 2. PyTorch도 중요

앞으로 딥러닝 공부에서는 TensorFlow/Keras뿐 아니라 PyTorch도 많이 사용하게 된다.

특히 다음 분야에서 PyTorch 사용 비중이 높다.

```text
RNN / LSTM / GRU
Transformer
LLM
HuggingFace
Computer Vision
Research Code
```

TensorFlow/Keras로 기본 개념을 익히면서 PyTorch 문법도 함께 익혀두는 것이 좋다.

---

## 3. SimpleRNN

RNN(Recurrent Neural Network)은 순서가 있는 데이터를 처리하기 위한 신경망이다.

대표적인 시계열 데이터:

```text
주가
기온
센서 데이터
음성
문장
시계열 로그
```

RNN의 핵심은 현재 입력뿐 아니라 이전 시점의 hidden state를 함께 사용한다는 것이다.

```text
x1 → RNN → h1
           ↓
x2 → RNN → h2
           ↓
x3 → RNN → h3
```

기본식:

\[
h_t = f(W_xx_t + W_hh_{t-1} + b)
\]

즉 현재 시점에서:

```text
현재 입력 x_t
+
이전 hidden state h_(t-1)
```

를 함께 계산한다.

RNN은 입력과 출력의 개수에 따라 다음처럼 나눌 수 있다.

```text
One to One
One to Many
Many to One
Many to Many
```

하지만 핵심은 동일하다.

> 이전 hidden state와 현재 input이 함께 연산된다.

---

## 4. SimpleRNN의 문제

입력 sequence가 너무 길어지면 Back Propagation Through Time(BPTT) 과정에서 gradient 문제가 생길 수 있다.

```text
Gradient Vanishing
→ gradient가 매우 작아짐

Gradient Exploding
→ gradient가 매우 커짐
```

쉽게 비유하면:

> 시험 문제를 아주 많이 풀고 마지막에 한꺼번에 채점하는데, 너무 오래전에 풀었던 문제는 왜 그렇게 풀었는지 기억이 안 나는 느낌.

즉 긴 시퀀스에서는 오래된 정보 학습이 어려워질 수 있다.

---

## 5. LSTM

LSTM(Long Short-Term Memory)은 SimpleRNN의 장기 기억 문제를 보완하기 위해 만들어졌다.

SimpleRNN은 hidden state 하나를 사용하지만 LSTM은 두 상태를 사용한다.

```text
h_t
→ short-term state

C_t
→ long-term state
```

즉:

```text
hidden state h_t
+
cell state C_t
```

를 함께 사용한다.

### LSTM의 3개 Gate

```text
Input Gate
→ 이번 입력을 얼마나 반영할지

Forget Gate
→ 과거 정보를 얼마나 버릴지

Output Gate
→ 이번 정보를 얼마나 밖으로 내보낼지
```

핵심 수식:

\[
f_t = \sigma(W_f[x_t,h_{t-1}] + b_f)
\]

\[
i_t = \sigma(W_i[x_t,h_{t-1}] + b_i)
\]

\[
	ilde C_t = 	anh(W_c[x_t,h_{t-1}] + b_c)
\]

\[
C_t = f_t \odot C_{t-1} + i_t \odot 	ilde C_t
\]

\[
o_t = \sigma(W_o[x_t,h_{t-1}] + b_o)
\]

\[
h_t = o_t \odot 	anh(C_t)
\]

여기서 `⊙`는 Hadamard Product, 즉 element-wise product이다.

---

## 6. GRU

GRU(Gated Recurrent Unit)는 LSTM을 더 단순하게 만든 구조다.

```text
LSTM
→ Gate 3개
→ h_t + C_t

GRU
→ Gate 2개
→ h_t만 사용
```

GRU에는 대표적으로 두 gate가 있다.

```text
Reset Gate r_t
Update Gate z_t
```

### Reset Gate

지난 정보를 얼마나 버릴지 결정한다.

\[
r_t = \sigma(W_rx_t + U_rh_{t-1} + b_r)
\]

### Update Gate

이전 정보를 얼마나 유지하고 현재 정보를 얼마나 반영할지 결정한다.

\[
z_t = \sigma(W_zx_t + U_zh_{t-1} + b_z)
\]

### Candidate Hidden State

\[
	ilde h_t =
	anh(
W_hx_t +
U_h(r_t \odot h_{t-1})
+
b_h
)
\]

### 최종 Hidden State

대표적인 표현:

\[
h_t =
(1-z_t)\odot h_{t-1}
+
z_t\odot 	ilde h_t
\]

즉 이전 기억 일부와 새로운 기억 일부를 섞어 새로운 hidden state를 만든다.

---

## 7. LSTM vs GRU

| 구분 | LSTM | GRU |
|---|---|---|
| Gate | 3개 | 2개 |
| Cell State | 있음 | 없음 |
| Hidden State | 있음 | 있음 |
| 구조 | 복잡 | 비교적 단순 |
| 파라미터 | 많음 | 상대적으로 적음 |
| 장기 의존성 처리 | 좋음 | 좋음 |

한 줄로 보면:

> LSTM은 기억을 더 세밀하게 관리하고, GRU는 비슷한 목적을 더 단순한 구조로 수행한다.

---

## 8. 시계열 데이터 자르기

RNN에서는 긴 시계열 데이터를 일정 길이로 잘라 여러 개의 학습 샘플로 만드는 작업을 많이 한다.

```python
def split_x(dataset, size):
    aaa = []

    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i+size]
        aaa.append(subset)

    return np.array(aaa)
```

예:

```python
dataset = np.array([1,2,3,4,5,6])
size = 3
```

결과:

```text
[1,2,3]
[2,3,4]
[3,4,5]
[4,5,6]
```

이 방식을 Sliding Window라고 한다.

### `len(dataset) - size + 1`

생성 가능한 window 개수다.

```text
dataset 길이 = 6
size = 3

6 - 3 + 1 = 4
```

### `dataset[i:i+size]`

현재 위치에서 `size`개만큼 데이터를 자른다.

```text
i=0
dataset[0:3]
→ [1,2,3]

i=1
dataset[1:4]
→ [2,3,4]
```

---

## 9. 2차원 데이터 자르기

예:

```python
dataset = np.array([
    [1,9],
    [2,8],
    [3,7],
    [4,6],
    [5,5]
])
```

`size=3`으로 자르면:

```text
[
 [[1,9],
  [2,8],
  [3,7]],

 [[2,8],
  [3,7],
  [4,6]],

 [[3,7],
  [4,6],
  [5,5]]
]
```

shape은:

```text
(samples, timesteps, features)
```

즉 바로 RNN 입력 형태가 된다.

---

## 10. split_x() 결과에서 x와 y 분리

```python
temp = split_x(dataset, size+1)

x = temp[:, :-1, :]
y = temp[:, -1, 1]
```

의미:

```text
:
→ 모든 sample

:-1
→ 마지막 timestep을 제외한 모든 timestep

:
→ 모든 feature
```

따라서:

```python
x = temp[:, :-1, :]
```

은 입력 시퀀스를 만들고,

```python
y = temp[:, -1, 1]
```

은 각 sample의 마지막 timestep에서 두 번째 feature를 정답으로 사용한다.

---

## 11. split_xy() 함수

x와 y를 처음부터 한 번에 만들 수도 있다.

```python
def split_xy(dataset, size):
    x = []
    y = []

    for i in range(len(dataset) - size):
        x_subset = dataset[i:i+size]
        y_subset = dataset[i+size][1]

        x.append(x_subset)
        y.append(y_subset)

    return np.array(x), np.array(y)
```

핵심:

```python
x_subset = dataset[i:i+size]
```

→ 과거 `size`개의 timestep

```python
y_subset = dataset[i+size][1]
```

→ 바로 다음 timestep의 두 번째 feature

---

## 12. 실습 데이터

```python
a = np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [9,8,7,6,5,4,3,2,1,0],
]).T
```

결과:

```text
[[ 1, 9],
 [ 2, 8],
 [ 3, 7],
 [ 4, 6],
 [ 5, 5],
 [ 6, 4],
 [ 7, 3],
 [ 8, 2],
 [ 9, 1],
 [10, 0]]
```

shape:

```text
(10, 2)
```

즉:

```text
10 timesteps
2 features
```

---

## 13. size=4로 split_xy()

```python
size = 4
x, y = split_xy(a, size)
```

결과 shape:

```python
x.shape
# (6, 4, 2)

y.shape
# (6,)
```

의미:

```text
6 = 학습 sample 수
4 = timestep
2 = feature
```

첫 번째 sample:

```text
x =
[[1,9],
 [2,8],
 [3,7],
 [4,6]]

y = 5
```

즉 앞의 4개 시점을 보고 다음 시점의 두 번째 feature 값을 예측한다.

---

## 14. GRU 모델

```python
model = Sequential()

model.add(GRU(30, input_shape=(4,2)))

model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
```

RNN 계열 입력은 기본적으로:

```text
(batch, timesteps, features)
```

현재 데이터는:

```text
(None, 4, 2)
```

즉 4 timestep, 2 feature를 입력받는다.

---

## 15. Learning Rate

```python
learning_rate = 0.008
```

Adam optimizer:

```python
optimizer = Adam(
    learning_rate=learning_rate
)
```

Learning Rate는:

> weight를 한 번 업데이트할 때 얼마나 큰 보폭으로 이동할지 정하는 값

이다.

---

## 16. Compile

이 문제는 숫자를 예측하는 회귀 문제이므로 `mse`가 적절하다.

```python
model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=learning_rate),
    metrics=['mae']
)
```

`accuracy`는 분류 문제에 적합하므로 회귀에서는 `mae`, `mse`, `rmse` 등을 보는 것이 더 자연스럽다.

---

## 17. Callback 사용 시 주의

```python
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=200,
    factor=0.5,
    verbose=1
)

es = EarlyStopping(
    monitor='val_loss',
    patience=500,
    restore_best_weights=True,
    verbose=1
)
```

Callback은 정의만 하면 동작하지 않는다.

반드시 `model.fit()`에 넣어야 한다.

```python
model.fit(
    x,
    y,
    epochs=1000,
    validation_split=0.2,
    callbacks=[rlr, es]
)
```

`val_loss`를 monitor하려면 validation 데이터가 있어야 한다.

---

## 18. Evaluate / Predict

```python
results = model.evaluate(x, y)
print('loss : ', results)
```

예측할 때도 학습 입력과 같은 timestep, feature 구조를 맞춰야 한다.

```python
x_predict = np.array([
    [7,3],
    [8,2],
    [9,1],
    [10,0]
]).reshape(1,4,2)

y_predict = model.predict(x_predict)

print('다음 예측의 결과:', y_predict)
```

shape:

```text
(1, 4, 2)

1 = 예측할 sample 수
4 = timestep
2 = feature
```

---

## 19. 전체 수정 코드

```python
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


a = np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [9,8,7,6,5,4,3,2,1,0],
]).T

size = 4


def split_xy(dataset, size):
    x = []
    y = []

    for i in range(len(dataset) - size):
        x_subset = dataset[i:i+size]
        y_subset = dataset[i+size, 1]

        x.append(x_subset)
        y.append(y_subset)

    return np.array(x), np.array(y)


x, y = split_xy(a, size)

print(x.shape)  # (6,4,2)
print(y.shape)  # (6,)


model = Sequential()

model.add(GRU(30, input_shape=(4,2)))
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))


learning_rate = 0.008

model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=learning_rate),
    metrics=['mae']
)


rlr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=200,
    factor=0.5,
    verbose=1
)

es = EarlyStopping(
    monitor='val_loss',
    patience=500,
    restore_best_weights=True,
    verbose=1
)


model.fit(
    x,
    y,
    epochs=1000,
    validation_split=0.2,
    callbacks=[rlr, es]
)


results = model.evaluate(x, y)
print('loss : ', results)


x_predict = np.array([
    [7,3],
    [8,2],
    [9,1],
    [10,0]
]).reshape(1,4,2)

y_predict = model.predict(x_predict)

print('다음 예측의 결과:', y_predict)
```
