# Day 14 - CNN 구조 설계, Padding / Stride / MaxPooling / GlobalAveragePooling

## 1. 오늘 학습 핵심

Day 14에서는 CNN 모델을 단순히 쌓는 것에서 한 단계 더 나아가,

- `padding`
- `stride`
- `MaxPooling2D`
- `Flatten`
- `GlobalAveragePooling2D`
- CNN과 DNN 비교
- 모델의 shape 변화
- 파라미터 수와 연산량의 차이
- 성능이 안 나올 때 모델 구조를 분석하는 방법

## 2. Padding

Padding은 Conv 연산 전에 이미지 가장자리에 값을 추가하는 방법이다. 보통 0을 채운다.

```python
Conv2D(
    64,
    (3,3),
    padding='same'
)
```

### `padding='valid'`

기본값이며 Padding을 넣지 않는다.

```text
32×32
↓ 3×3 Conv
30×30
```

### `padding='same'`

`stride=1`일 때 출력의 가로/세로 크기를 유지한다.

```text
32×32
↓ 3×3 Conv, same
32×32
```

사용 이유:

```text
1. Feature Map이 너무 빨리 작아지는 것을 방지
2. 이미지 가장자리 정보도 더 활용
```

---

## 3. Stride

Stride는 커널이 몇 칸씩 이동할지를 의미한다.

```text
stride=1
→ 한 칸씩 이동

stride=2
→ 두 칸씩 이동
```

stride가 커지면:

```text
Feature Map 크기 감소
연산량 감소
세밀한 공간 정보 손실 가능
```

예:

```python
Conv2D(
    128,
    (3,3),
    strides=2,
    padding='same'
)
```

대략:

```text
32×32
↓
16×16
```

처럼 줄어든다.

---

## 4. MaxPooling

`MaxPooling2D`는 작은 영역에서 가장 큰 값 하나만 남긴다.

예:

```text
1 3
4 6
```

이면:

```text
6
```

을 남긴다.

```python
MaxPooling2D(pool_size=(2,2))
```

는 2×2 영역마다 최대값 하나를 선택한다.

보통:

```text
28×28
↓
14×14
```

처럼 가로/세로 크기가 절반 정도로 줄어든다.

역할:

```text
Conv2D
→ 특징 추출

MaxPooling
→ 강하게 나타난 특징을 남기면서 크기 축소
```

---

## 5. Conv / Dropout / MaxPooling 순서

실습에서는 보통 이런 구조를 자주 본다.

```text
Conv
↓
Conv
↓
MaxPooling
↓
Dropout
```

또는:

```text
Conv
↓
Dropout
↓
MaxPooling
```

도 가능하다.

절대적인 정답 순서는 아니고, 중요한 것은 각 역할을 알고 배치하는 것이다.

```text
Conv → 특징 추출
Pooling → 공간 축소
Dropout → 과적합 완화
```

---

## 6. CNN Parameter란?

`model.summary()`의 `Param #`은 연산량이 아니다.

> 학습해야 하는 Weight와 Bias의 개수이다.

Conv2D 파라미터 공식:

```text
(kernel 높이 × kernel 너비 × 입력 채널 + bias 1개)
× filter 개수
```

예:

```text
3×3 kernel
입력 channel = 3
filter = 64
```

이면:

```text
(3×3×3 + 1) × 64
= 1,792
```

즉:

```text
Parameter 수 ≠ 연산 횟수
```

이다.

---

## 8. 지금까지 모델의 문제점 분석

이 구조에서는:

```text
stride=2
MaxPooling
stride=2
stride=2
MaxPooling
```

처럼 다운샘플링이 여러 번 일어난다.

따라서:

```text
32
↓
16
↓
8
↓
4
↓
2
↓
1
```

처럼 Feature Map이 너무 빨리 줄어들 수 있다.

문제:

```text
공간 정보가 너무 빨리 사라짐
작은 이미지에서는 정보 손실이 큼
뒤쪽 Conv에서 사용할 공간 정보가 부족해짐
```

CIFAR는 원래 32×32로 작은 이미지이므로 `stride=2`와 `MaxPooling`을 너무 많이 같이 쓰지 않는 것이 중요하다.

---

## 9. 좋은 CNN 구조를 볼 때 중요한 기준

일반적으로 많이 보는 흐름:

```text
공간 크기 ↓
채널 수 ↑
```

예:

```text
32×32×32
↓
16×16×64
↓
8×8×128
↓
4×4×256
```

초반에는 공간 정보를 더 유지하고, 뒤로 갈수록 더 다양한 특징을 표현하는 방향이다.

---

## 10. Flatten의 문제

예를 들어 CNN 마지막 출력이:

```text
100 × 100 × 64
```

라면:

```python
Flatten()
```

후:

```text
100 × 100 × 64
= 640,000
```

개의 값이 된다.

즉:

```text
(N, 100, 100, 64)
↓ Flatten
(N, 640000)
```

이다.

여기에 Dense를 연결하면 파라미터 수가 매우 커질 수 있다.

예:

```text
640000 → Dense(128)
```

이면 대략:

```text
640000 × 128
```

개의 Weight가 필요하다.

따라서:

```text
파라미터 폭증
학습시간 증가
메모리 사용량 증가
과적합 위험 증가
```

가 발생할 수 있다.

---

## 11. GlobalAveragePooling2D

Flatten 대신 사용할 수 있는 대표적인 방법 중 하나이다.

```python
from tensorflow.keras.layers import GlobalAveragePooling2D
```

예:

```text
(N, 100, 100, 64)
```

입력에서 각 채널의 `100×100` 값을 모두 평균낸다.

```text
Channel 1의 전체 평균 → 숫자 1개
Channel 2의 전체 평균 → 숫자 1개
...
Channel 64의 전체 평균 → 숫자 1개
```

결과:

```text
(N, 100, 100, 64)
↓ GlobalAveragePooling2D
(N, 64)
```

즉:

```text
Flatten
→ 640000개

GlobalAveragePooling
→ 64개
```

로 크게 줄어든다.

핵심:

> `GlobalAveragePooling2D`는 단순히 shape만 바꾸는 것이 아니라, 각 Feature Map 전체 평균을 계산해 채널마다 숫자 하나만 남긴다.

---

## 12. AveragePooling과 GlobalAveragePooling 차이

### AveragePooling2D

작은 구역별 평균을 계산한다.

```text
4×4
↓ AveragePooling(2×2)
2×2
```

아직 4차원 Feature Map이다.

### GlobalAveragePooling2D

채널 하나의 전체 공간 평균을 계산한다.

```text
7×7×64
↓
64
```

즉:

```text
(batch, height, width, channel)
↓
(batch, channel)
```

로 바로 2차원이 된다.

---

## 13. GlobalAveragePooling을 사용하는 이유

장점:

```text
Flatten 대비 파라미터 수 감소
메모리 감소
연산량 감소
과적합 완화 가능
모델 구조 단순화
```

단점:

```text
위치 정보가 많이 사라짐
```

따라서 이미지 분류에서는 잘 맞을 수 있지만, 위치 정보가 중요한 Segmentation에서는 무조건 쓰는 것은 아니다.

---

## 14. 이미지 데이터라고 CNN만 사용하는 것은 아니다

이미지 데이터도 DNN으로 학습할 수 있다.

예:

```text
28×28 이미지
↓
reshape
784
↓
Dense
```

MNIST:

```python
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
```

DNN 예시:

```python
model = Sequential()

model.add(Dense(128, input_shape=(784,), activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))
```

다만 DNN은 이미지를 일렬로 펼치기 때문에 픽셀 간 공간 관계를 직접 활용하기 어렵다.

CNN은 공간 구조를 유지하면서 특징을 찾는 것이 장점이다.

---

## 15. 반대로 일반 표 데이터를 CNN으로 바꿀 수도 있다

예를 들어:

```text
(samples, 30)
```

형태의 데이터를:

```python
x_train = x_train.reshape(-1, 5, 6, 1)
x_test = x_test.reshape(-1, 5, 6, 1)
```

처럼 4차원으로 만들어 Conv2D에 넣을 수는 있다.

```python
model = Sequential()

model.add(
    Conv2D(
        64,
        (2,1),
        input_shape=(5,6,1),
        activation='relu',
        padding='same'
    )
)

model.add(
    Conv2D(
        32,
        (2,1),
        activation='relu',
        padding='same'
    )
)

model.add(Dropout(0.5))

model.add(Conv2D(30, (2,1), activation='relu', padding='same'))
model.add(Conv2D(30, (2,1), activation='relu', padding='same'))
model.add(Conv2D(30, (2,1), activation='relu', padding='same'))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

다만 중요한 점:

> 일반적인 표 데이터는 실제 이미지처럼 2차원 공간 관계가 없기 때문에, 억지로 reshape했다고 CNN이 반드시 더 좋아지는 것은 아니다.

따라서 DNN과 CNN을 둘 다 돌려 성능을 비교해보는 실험은 의미가 있다.

---

## 17. DNN vs CNN 비교 실험

오늘 실습에서는 기존 DNN 모델들을 CNN 구조로 바꾸고 reshape를 적용한 뒤 성능을 비교했다.

비교할 항목:

```text
Train Accuracy
Validation Accuracy
Test Accuracy
Loss
Parameter 수
학습시간
```

중요한 것은:

> CNN이 항상 DNN보다 좋다는 결론을 내리는 것이 아니라, 어떤 데이터에서 어떤 구조가 더 잘 맞는지 비교하는 것이다.

---

## 18. 성능이 안 나올 때 분석할 것

모델을 무작정 크게 만들기 전에 아래를 확인한다.

```text
1. 입력 shape가 맞는가?
2. Scaling이 되었는가?
3. Output unit 수가 맞는가?
4. Loss가 문제 종류와 맞는가?
5. Feature Map이 너무 빨리 작아지지 않는가?
6. stride=2를 너무 많이 쓰지 않았는가?
7. MaxPooling을 너무 많이 쓰지 않았는가?
8. Flatten 이후 Parameter가 폭증하지 않는가?
9. Dropout이 너무 강하지 않은가?
10. Train과 Validation 성능 차이가 큰가?
```

```text
코드를 실행하는 능력
↓
구조를 이해하는 능력
↓
문제를 진단하는 능력
↓
실험을 설계하는 능력
↓
AI 시스템을 만드는 능력
```

으로 발전하는 것이 목표이다.

---

## 20. YOLO / Segmentation과 CNN

CNN에서 배우는:

```text
Conv2D
Padding
Stride
Pooling
Feature Map
```

개념은 이후 Object Detection과 Segmentation의 기본이 된다.

### YOLO

```text
Image
↓
Backbone
↓
Feature Extraction
↓
Neck
↓
Detection Head
↓
Bounding Box + Class
```

### Segmentation

대표적인 모델/계열:

```text
U-Net
DeepLab
SegFormer
SAM 계열
```

SAM 계열은 범용적인 prompt 기반 segmentation 모델이다.

---