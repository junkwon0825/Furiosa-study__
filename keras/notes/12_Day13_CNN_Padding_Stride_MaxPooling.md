# Day 13 - CNN 심화: Reshape, Flatten, Padding, Stride, MaxPooling

## 1. 오늘 학습 핵심

Day 13에서는 CNN에서 이미지 차원을 다루는 방법과 `Conv2D`, `Flatten`, `Padding`, `Stride`, `MaxPooling`의 역할을 학습했다.

```text
CNN 입력 차원
↓
Conv2D 반복
↓
Feature Map 변화
↓
Flatten
↓
Dense 분류
↓
Padding
↓
Stride
↓
MaxPooling
```

---

## 2. CNN은 이미지를 어떻게 보는가?

CNN은 작은 `Kernel(Filter)`을 이미지 위에서 이동시키며 공간적인 특징을 찾는다.

```text
Image
↓
Conv2D
↓
Edge / Line / Color 변화
↓
더 복잡한 Shape
↓
Object 특징
↓
Classification
```

초기 Conv 층에서는 선, 경계, 색 변화 같은 단순한 특징을 찾고, 층이 깊어질수록 더 복잡한 패턴을 표현한다.

하지만 Conv2D를 많이 쌓는다고 무조건 성능이 좋아지는 것은 아니다.

```text
공간 크기 감소
연산량 증가
학습 시간 증가
과적합 가능성 증가
정보 손실 가능
```

때문에 적절한 깊이를 찾아야 한다.

---

## 3. CNN 입력 Shape

Keras `Conv2D`의 일반적인 입력 형태:

```text
(batch, height, width, channels)
```

예:

```text
(60000, 28, 28, 1)
```

의미:

```text
60000 = 이미지 개수
28    = 높이(height)
28    = 너비(width)
1     = 채널(channel)
```

`input_shape`에서는 batch를 제외한다.

```python
input_shape=(28,28,1)
```
즉:

```text
(height, width, channels)
```
이다.

예:

```python
input_shape=(5,5,1)
```

```text
5 = height
5 = width
1 = channel
```

흑백은 보통 channel=1, RGB 컬러는 channel=3이다.

---

## 4. MNIST와 CIFAR의 차원

### MNIST

처음 불러오면:

```text
(60000, 28, 28)
```

즉 3차원이다.

Conv2D에 넣으려면 채널 축을 추가한다.

```python
x_train = x_train.reshape(-1,28,28,1)
```

결과:

```text
(60000, 28, 28, 1)
```

### CIFAR-10 / CIFAR-100

처음부터:

```text
(50000, 32, 32, 3)
```

형태이므로 이미 4차원이다.

따라서 보통 추가 reshape가 필요 없다.

---

## 5. OneHotEncoder와 reshape

분류 문제에서 정수 라벨을 one-hot 형태로 바꿀 수 있다.

예:

```text
2
↓
[0, 0, 1, 0, ...]
```

Scikit-learn `OneHotEncoder`는 보통 2차원 입력을 기대하므로:

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

y = y.reshape(-1,1)
y = ohe.fit_transform(y)
```

처럼 사용한다.

예:

```text
(150,)
↓ reshape
(150,1)
↓ OneHotEncoder
(150,3)
```

`sparse_output=False`는 결과를 희소행렬이 아니라 일반적인 dense NumPy 배열 형태로 반환한다는 뜻이다.

혼동행렬(confusion matrix)과는 관계가 없다.

또한 `fit_transform()`은 y 전용 함수가 아니다.

예를 들어 X Scaling에서도:

```python
x_train = scaler.fit_transform(x_train)
```

처럼 사용한다.

즉:

```text
fit_transform()
= fit + transform
```

이다.

---

## 6. Conv2D Output Shape

기본적으로:

```text
padding='valid'
stride=1
```

이면 출력 크기:

```text
Output = Input - Kernel + 1
```

예:

```text
Input = 28
Kernel = 3
```

이면:

```text
28 - 3 + 1 = 26
```

따라서:

```text
28 × 28
↓ Conv2D(3×3)
26 × 26
```

가 된다.

---

## 7. Conv2D Parameter 계산

공식:

```text
(Kernel_H × Kernel_W × Input_Channel + 1) × Filter_Count
```

`+1`은 filter마다 존재하는 bias이다.

예:

```text
Input Channel = 3
Kernel = 3×3
Filter = 64
```

이면:

```text
(3×3×3 + 1) × 64
= 1,792
```

---

## 8. Flatten

Conv2D 출력은 보통 4차원 형태이다.

예:

```text
(None, 17, 17, 30)
```

Dense 층으로 연결하기 위해 1차원 벡터 형태로 펼친다.

```python
model.add(Flatten())
```

계산:

```text
17 × 17 × 30
= 8670
```

따라서:

```text
(None, 17, 17, 30)
↓ Flatten
(None, 8670)
```

이 된다.

Flatten은 값 자체를 학습하지 않고 모양만 바꾸므로:

```text
Param = 0
```
이다.

---


## 10. Padding

Padding은 Conv 연산 전에 이미지 가장자리에 값을 추가하는 것이다.

보통 0을 추가한다.

예:

```text
원본

1 2 3
4 5 6
7 8 9
```

Zero Padding:

```text
0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
```

### padding='valid'

Padding을 넣지 않는다.

```python
padding='valid'
```

기본값이다.

예:

```text
32×32
↓ Conv 3×3
30×30
```

### padding='same'

출력 공간 크기가 급격히 줄어드는 것을 막기 위해 사용한다.

```python
padding='same'
```

stride=1이라면:

```text
32×32
↓ Conv
32×32
```

처럼 크기가 유지된다.

Padding을 쓰는 이유:

```text
1. 공간 크기가 너무 빨리 줄어드는 것을 방지
2. 이미지 가장자리 정보도 더 잘 활용
```

---

## 11. Padding에서 중요한 예외: stride > 1

`padding='same'`이라고 해서 항상 입력 크기와 완전히 같은 것은 아니다.

stride=1일 때 주로 같은 크기를 유지한다.

예:

```python
Conv2D(
    10,
    (2,2),
    input_shape=(10,10,1),
    strides=2,
    padding='same'
)
```

여기서는 stride가 2이므로:

```text
10×10
↓ stride=2
5×5
```

정도가 된다.

즉 출력은:

```text
(None, 5, 5, 10)
```

이다.

---

## 12. Stride

Stride는 Kernel이 움직이는 간격이다.

기본값:

```text
stride = 1
```

이면 한 칸씩 이동한다.

`stride=2`이면 두 칸씩 이동한다.

Stride가 커질수록:

```text
출력 Feature Map 크기 감소
연산량 감소
세밀한 정보 손실 가능
```

이 있다.

---

## 13. Padding + Stride 예제

```python
model = Sequential()

model.add(
    Conv2D(
        10,
        (2,2),
        input_shape=(10,10,1),
        strides=2,
        padding='same'
    )
)

model.add(
    Conv2D(
        filters=9,
        kernel_size=(3,3),
        strides=1,
        padding='valid'
    )
)

model.summary()
```

첫 번째 Conv:

```text
10×10×1
↓
stride=2, padding='same'
↓
5×5×10
```

두 번째 Conv:

```text
5×5×10
↓
3×3 kernel, valid
↓
3×3×9
```

---

## 14. MaxPooling

MaxPooling은 CNN의 Feature Map 크기를 줄이면서 강하게 나타난 특징을 남기는 연산이다.

예:

```text
1 3
4 6
```

에서:

```text
max = 6
```

만 남긴다.

### MaxPooling2D((2,2))

예:

```text
4×4 Feature Map
↓
2×2 영역마다 최대값 선택
↓
2×2 Feature Map
```

예:

```text
1 3 2 0
4 6 1 2
5 2 8 1
0 1 3 7
```

2×2 MaxPooling 결과:

```text
6 2
5 8
```

---

## 15. MaxPooling의 역할

```text
Conv2D
→ 특징을 찾음

MaxPooling
→ 강한 특징을 남기면서 공간 크기를 줄임
```

전체 흐름:

```text
Image
↓
Conv2D
↓
Edge / Line / Color 변화
↓
Feature Map
↓
MaxPooling
↓
중요한 특징은 유지
↓
크기는 축소
↓
다음 Conv
```

---

## 16. MaxPooling의 장점

### 1. 연산량 감소

예:

```text
32×32×64
```

에서:

```python
MaxPooling2D((2,2))
```

하면 보통:

```text
16×16×64
```

가 된다.

값의 개수:

```text
32×32×64 = 65,536
↓
16×16×64 = 16,384
```

즉 1/4 수준으로 줄어든다.

### 2. 강한 특징 유지

특정 영역에서 가장 크게 반응한 값을 남겨 특징 존재 여부를 요약한다.

### 3. Parameter 없음

MaxPooling은 학습할 weight가 없다.

```text
Param = 0
```

---

## 17. Conv2D / Padding / Stride / MaxPooling 역할 비교

| 기능 | 역할 |
|---|---|
| Conv2D | 특징 추출 |
| Padding | 가장자리 보존 / 크기 감소 조절 |
| Stride | Kernel 이동 간격 조절 |
| MaxPooling | 특징맵 축소 + 강한 특징 유지 |
| Flatten | CNN 출력을 Dense에 넣기 위해 1차원으로 펼침 |

---


## 18. 오늘 공부한 데이터셋

### MNIST

```text
28×28×1
10 classes
```

### Fashion MNIST

```text
28×28×1
10 classes
```

### CIFAR-10

```text
32×32×3
10 classes
```

### CIFAR-100

```text
32×32×3
100 classes
```

특히 CIFAR 데이터는 RGB 이미지이므로:

```text
channel = 3
```

이다.

---