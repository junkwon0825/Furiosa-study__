# DAY11 딥러닝 학습 정리

## 1. 모델과 가중치 저장

딥러닝 모델을 학습한 뒤 결과를 저장하는 방법은 크게 두 가지가 있다.

```text
1. 전체 모델 저장
2. 가중치만 저장
```

---

## 2. 전체 모델 저장

```python
model.save("model.keras")
```

전체 모델을 저장하면 일반적으로 다음 정보가 포함된다.

```text
Model Architecture
+
Weights
+
일부 학습 설정
```

불러오기:

```python
from tensorflow.keras.models import load_model

model = load_model("model.keras")
```

이 방식은 모델 구조를 다시 작성하지 않아도 된다는 장점이 있다.

---

## 3. 가중치만 저장

```python
model.save_weights("model.weights.h5")
```

이 경우 모델의 가중치만 저장한다.

따라서 불러올 때는 먼저 **동일한 모델 구조**가 필요하다.

```python
model = Sequential()

model.add(Dense(...))
model.add(Dense(...))
model.add(Dense(...))

model.load_weights("model.weights.h5")
```

### 전체 모델 저장 vs 가중치 저장

| 방법 | 모델 구조 다시 작성 필요 | Weight 저장 |
|---|---:|---:|
| `model.save()` | X | O |
| `model.save_weights()` | O | O |

모델 구조는 코드로 관리하고 가중치만 따로 저장하고 싶을 때 `save_weights()`를 사용할 수 있다.

---

# 4. 학습 중 가중치는 계속 바뀐다

학습 과정에서는 Epoch마다 Weight가 계속 업데이트된다.

```text
Epoch 1
Weight A

↓

Epoch 2
Weight B

↓

Epoch 3
Weight C
```

학습이 끝난 뒤 저장한 가중치를 불러오면:

```python
model.load_weights(...)
```

다시 학습하지 않는 한 저장된 Weight를 그대로 사용해서 평가와 예측을 할 수 있다.

---

# 5. EarlyStopping

```python
from tensorflow.keras.callbacks import EarlyStopping
```

사용 예:

```python
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1
)
```

`EarlyStopping`은 검증 성능이 더 이상 좋아지지 않을 때 학습을 자동으로 종료하는 기능이다.

---

## monitor

```python
monitor='val_loss'
```

어떤 값을 관찰할 것인지 지정한다.

```text
val_loss를 계속 확인
```

---

## mode

```python
mode='min'
```

`val_loss`는 낮을수록 좋기 때문에:

```text
min
```

을 사용한다.

---

## patience

```python
patience=20
```

20 Epoch 동안 성능 개선이 없으면 학습을 종료한다.

예:

```text
Epoch 30 → best
Epoch 31 → 개선 없음
Epoch 32 → 개선 없음
...
Epoch 50 → 개선 없음

→ 학습 종료
```

---

## restore_best_weights

```python
restore_best_weights=True
```

학습 종료 후 가장 성능이 좋았던 Epoch의 Weight로 복원한다.

---

# 6. ModelCheckpoint

```python
from tensorflow.keras.callbacks import ModelCheckpoint
```

예:

```python
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath='best_model.keras',
    verbose=1
)
```

훈련 중 모델을 파일로 저장한다.

특히:

```python
save_best_only=True
```

로 설정하면 `val_loss`가 개선된 모델만 저장한다.

출력 예:

```text
val_loss improved from 0.28269 to 0.28247,
saving model to best_model.keras
```

---

# 7. EarlyStopping + ModelCheckpoint

둘은 같이 자주 사용한다.

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(
    loss='mse',
    optimizer='adam'
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath='best_model.keras',
    verbose=1
)

history = model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=40,
    validation_split=0.15,
    callbacks=[es, mcp],
    verbose=1
)
```

역할:

```text
EarlyStopping
→ 더 이상 성능이 좋아지지 않으면 학습 종료

ModelCheckpoint
→ 학습 중 좋은 모델을 저장
```

---

# 8. Epoch 정보와 val_loss를 파일명에 넣기

```python
import datetime

date = datetime.datetime.now()

print(type(date))

date = date.strftime("%m%d_%H%M")

print(date)
print(type(date))
```

예:

```text
0914_1148
```

파일명 만들기:

```python
path = './_save/keras31/'

filename = '-{epoch:04d}-{val_loss:.4f}.keras'

filepath = "".join([
    path,
    "k31_",
    date,
    filename
])
```

생성 예:

```text
k31_0914_1148-0001-0.3124.keras
k31_0914_1148-0002-0.2981.keras
k31_0914_1148-0003-0.2842.keras
```

의미:

```text
k31_
→ 프로젝트 이름

0914_1148
→ 날짜 / 시간

0003
→ Epoch

0.2842
→ val_loss
```

---

# 9. ModelCheckpoint에 파일명 적용

```python
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=filepath,
    verbose=1
)
```

주의:

```python
save_best_only=True
```

이면 **모든 Epoch가 저장되는 것이 아니라 성능이 개선된 Epoch만 저장된다.**

모든 Epoch를 저장하려면:

```python
save_best_only=False
```

를 사용할 수 있다.

---

# 10. 저장한 Weight 불러오기

```python
path = './_save/keras30/'

model.load_weights(
    path + '저장한_weights_파일명'
)
```

불러온 뒤 바로 평가와 예측이 가능하다.

```python
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
```

---

# 11. Dropout

## Dropout이란?

`Dropout`은 학습할 때 일부 뉴런의 출력을 **랜덤하게 0으로 만드는 기법**이다.

예:

```python
model.add(Dropout(0.2))
```

의 의미:

```text
학습 중 일부 뉴런을 랜덤하게 비활성화
```

대략 20% 비율로 Drop한다.

---

## 왜 Dropout을 사용하는가?

모델이 특정 뉴런 몇 개에 너무 의존하면 과적합이 발생하기 쉽다.

```text
Train Data에서는 성능 좋음
        ↓
새로운 Data에서는 성능 하락
        ↓
Overfitting
```

Dropout을 적용하면 학습할 때마다 일부 뉴런이 랜덤하게 제외된다.

```text
학습 A
Neuron 1 ON
Neuron 2 OFF
Neuron 3 ON

학습 B
Neuron 1 OFF
Neuron 2 ON
Neuron 3 ON
```

특정 뉴런 하나에 지나치게 의존하기 어려워지고 여러 뉴런이 다양한 특징을 학습하게 된다.

따라서:

```text
Overfitting 감소
↓
일반화 성능 향상 가능
```

---

# 12. 학습과 추론에서 Dropout 차이

### Training

```text
일부 뉴런 랜덤 비활성화
```

### Evaluation / Prediction

```text
Dropout 비활성화
모든 뉴런 사용
```

즉:

```python
model.fit()
```

에서는 Dropout이 동작하고,

```python
model.evaluate()
model.predict()
```

에서는 모든 뉴런을 사용한다.

---

# 13. Dropout 사용 예제

```python
model = Sequential()

model.add(Dense(256, input_dim=8))

model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(64))

model.add(Dense(32))

model.add(Dropout(0.5))

model.add(Dense(8, activation='relu'))

model.add(Dense(4))

model.add(Dense(1))
```

구조:

```text
Input(8)
  ↓
Dense(256)
  ↓
Dropout(0.2)
  ↓
Dense(128, ReLU)
  ↓
Dropout(0.3)
  ↓
Dense(64)
  ↓
Dense(32)
  ↓
Dropout(0.5)
  ↓
Dense(8, ReLU)
  ↓
Dense(4)
  ↓
Dense(1)
```

Dropout은 Dense Layer 사이에 넣을 수 있다.

---

# 14. Sequential 모델

Sequential 모델은 Layer를 순서대로 쌓는 방식이다.

```python
model = Sequential()

model.add(Dense(400, input_dim=64, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
```

구조:

```text
Input
 ↓
Layer
 ↓
Layer
 ↓
Layer
 ↓
Output
```

단순한 직선형 모델을 만들 때 편하다.

---

# 15. 함수형 모델 Functional API

```python
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

input1 = Input(shape=(64,))

dense1 = Dense(400, activation='relu')(input1)
dense2 = Dense(250, activation='relu')(dense1)

drop1 = Dropout(0.2)(dense2)

dense3 = Dense(100)(drop1)
dense4 = Dense(50)(dense3)

drop2 = Dropout(0.2)(dense4)

dense5 = Dense(40, activation='relu')(drop2)
dense6 = Dense(20, activation='relu')(dense5)

drop3 = Dropout(0.2)(dense6)

output1 = Dense(10, activation='softmax')(drop3)

model = Model(
    inputs=input1,
    outputs=output1
)
```

함수형 모델에서는 이전 Layer의 출력을 다음 Layer의 입력으로 넘긴다.

예:

```python
dense1 = Dense(400)(input1)

dense2 = Dense(250)(dense1)
```

의미:

```text
input1
  ↓
Dense(400)
  ↓
dense1
  ↓
Dense(250)
  ↓
dense2
```

---

# 16. Input 정의

Sequential:

```python
Dense(400, input_dim=64)
```

Functional API:

```python
input1 = Input(shape=(64,))
dense1 = Dense(400)(input1)
```

즉:

```text
input_dim=64
```

를 함수형 모델에서는:

```python
Input(shape=(64,))
```

로 따로 정의한다.

---

# 17. 함수형 모델의 장점

Sequential:

```text
Input
 ↓
A
 ↓
B
 ↓
C
 ↓
Output
```

처럼 한 줄 구조에 적합하다.

Functional API는:

```text
          → A →
Input                 → Output
          → B →
```

같은 복잡한 구조도 만들 수 있다.

예:

```text
입력 여러 개
출력 여러 개
Layer 분기
Layer 병합
복잡한 네트워크
```

그래서 복잡한 딥러닝 모델에서는 함수형 모델을 많이 사용한다.

---

# DAY11 핵심 정리

## 모델 저장

```text
model.save()
→ 전체 모델 저장

model.save_weights()
→ 가중치만 저장
→ 불러올 때 동일한 모델 구조 필요
```

---

## EarlyStopping

```text
val_loss 개선이 멈추면
→ 학습 자동 종료
```

---

## ModelCheckpoint

```text
학습 중 좋은 모델을 저장
```

대표 조합:

```text
EarlyStopping
+
ModelCheckpoint
```

---

## Dropout

```text
학습 중
→ 일부 뉴런 랜덤 비활성화

평가 / 추론
→ 모든 뉴런 사용
```

목적:

```text
Overfitting 감소
Generalization 향상
```

---

## Functional API

Sequential:

```python
model.add(Dense(...))
model.add(Dense(...))
```

Functional:

```python
input1 = Input(...)

dense1 = Dense(...)(input1)
dense2 = Dense(...)(dense1)

model = Model(
    inputs=input1,
    outputs=dense2
)
```

핵심:

> 이전 Layer의 출력이 다음 Layer의 입력이 된다.

-