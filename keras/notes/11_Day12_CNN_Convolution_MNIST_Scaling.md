# Day 12 - CNN, Convolution, MNIST, Image Scaling

## 1. 오늘 학습 핵심

Day 12에서는 **CNN(Convolutional Neural Network)** 의 기본 구조와 이미지 데이터의 형태,
`Conv2D`의 연산 방식, MNIST 데이터셋, 이미지 Scaling을 학습했다.

```text
CPU / GPU 차이
↓
DNN / RNN / CNN 비교
↓
이미지 데이터 구조
↓
Convolution(합성곱)
↓
Conv2D
↓
Output Shape / Parameter 계산
↓
MNIST
↓
Image Scaling
```

---

## 2. 전용 GPU 메모리와 공유 GPU 메모리

### 전용 GPU 메모리

**그래픽카드 자체에 붙어 있는 VRAM**이다.

```text
GPU ↔ VRAM
```

GPU가 직접 빠르게 사용할 수 있기 때문에 딥러닝에서 매우 중요하다.

주로 다음과 같은 정보가 올라간다.

```text
Model Weight
Activation
Gradient
Batch Data
```

### 공유 GPU 메모리

GPU가 필요할 때 **시스템 RAM을 빌려 사용하는 것**이다.

```text
GPU
↓
System Memory
↓
RAM
```

전용 VRAM보다 일반적으로 느리다.

> 딥러닝에서는 공유 GPU 메모리보다 **전용 GPU 메모리(VRAM)** 용량이 더 중요하다.

---

## 3. GPU는 언제 빠른가?

GPU는 **많은 계산을 동시에 병렬 처리하는 것**에 강하다.

```text
CPU에 있는 데이터
↓
GPU 메모리로 전달
↓
GPU 연산
↓
결과 반환
```

데이터와 모델이 매우 작으면 GPU를 사용하기 위한 준비 비용 때문에
오히려 CPU가 더 빠를 수도 있다.

### CPU가 유리할 수 있는 경우

```text
작은 데이터
작은 Dense 모델
적은 연산량
```

예:

```text
Boston
Diabetes
Breast Cancer
```

### GPU가 유리한 경우

```text
큰 데이터
큰 모델
큰 행렬 연산
CNN
Transformer
LLM
```

특히 이미지와 LLM처럼 연산량이 많아질수록 GPU 효과가 커진다.

> GPU는 무조건 빠른 장치가 아니라, **큰 병렬 연산이 많을수록 강한 장치**이다.

---

## 4. DNN / CNN / RNN 비교

| 모델 | 핵심 | 주로 사용하는 데이터 | 대표 예 |
| --- | --- | --- | --- |
| DNN | Dense 연결 | 표 형태 데이터 | 집값, 암 진단 |
| CNN | 공간 특징 추출 | 이미지 | 차선, 객체 탐지 |
| RNN | 이전 정보 기억 | 순서/시계열 | 문장, 센서 데이터 |

---

## 5. 일반적인 입력 데이터 차원

### DNN

```text
(samples, features)
```

예:

```text
(569, 30)
```

보통 2차원 입력을 사용한다.

### RNN

```text
(samples, timesteps, features)
```

예:

```text
(1000, 30, 5)
```

의미:

```text
1000개 샘플
30개 시간 구간
시간마다 feature 5개
```

보통 3차원 입력을 사용한다.

### CNN

Keras의 `Conv2D`는 일반적으로:

```text
(samples, height, width, channels)
```

형태의 **4차원 입력**을 사용한다.

예:

```text
(60000, 28, 28, 1)
```

의미:

```text
60000 = 이미지 개수
28    = 높이
28    = 너비
1     = 채널(흑백)
```

RGB 이미지라면 channel은 보통 `3`이다.

---

## 6. CNN이 이미지에서 하는 일

CNN은 이미지에서 공간적인 특징을 찾는다.

```text
Image
↓
Convolution
↓
Edge / Line
↓
Shape
↓
Higher-level Feature
↓
Classification
```

초기 층에서는 선이나 경계 같은 단순한 특징을 찾고,
층이 깊어질수록 더 복잡한 특징을 표현할 수 있다.

---

## 7. Convolution이란?

`Convolution`은 우리말로 **합성곱**이라고 한다.

작은 크기의 **Filter(Kernel)** 를 이미지 위에서 일정 간격으로 이동시키면서
해당 영역과 곱셈/덧셈 연산을 수행한다.

예:

입력 일부:

```text
1 2
3 4
```

Kernel:

```text
1 0
0 1
```

계산:

```text
1×1 + 2×0 + 3×0 + 4×1
= 5
```

이 값이 새로운 Feature Map의 한 값이 된다.

> Kernel을 이미지 위에서 이동시키면서 각 위치의 특징을 추출하는 것이 Convolution이다.

---

## 8. Conv2D는 이미지를 단순히 자르는 것이 아니다

초보 단계에서는 "커널 크기만큼 이미지를 본다"라고 이해하면 좋다.

정확히는:

```text
이미지를 잘라서 버리는 것 ❌
Kernel이 작은 영역을 보면서 연산하는 것 ✅
```

예:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

model = Sequential()

model.add(
    Conv2D(
        10,
        (2, 2),
        input_shape=(5, 5, 1)
    )
)

model.add(
    Conv2D(
        5,
        (2, 2)
    )
)
```

여기서:

```text
10      = Filter 개수
(2, 2)  = Kernel 크기
(5,5,1) = 입력 이미지 크기
```

---

## 9. Conv2D Output Shape 계산

기본값:

```text
padding = 'valid'
stride = 1
```

일 때:

```text
Output = Input - Kernel + 1
```

첫 번째 Conv2D:

```text
입력 : 5 × 5 × 1
Kernel : 2 × 2
Filter : 10
```

공간 크기:

```text
5 - 2 + 1 = 4
```

따라서:

```text
(None, 4, 4, 10)
```

`None`은 batch size가 아직 정해지지 않았다는 뜻이다.

---

## 10. Conv2D Parameter 계산

공식:

```text
(Kernel_H × Kernel_W × Input_Channel + Bias) × Filter_Count
```

Bias는 Filter마다 1개씩 있다.

### 첫 번째 Conv2D

```text
Kernel = 2 × 2
Input Channel = 1
Filter = 10
```

계산:

```text
(2 × 2 × 1 + 1) × 10
= 50
```

따라서:

```text
Param # = 50
```

### 두 번째 Conv2D

첫 번째 층 출력:

```text
4 × 4 × 10
```

따라서 두 번째 Conv2D의 Input Channel은 `10`이다.

```text
Kernel = 2 × 2
Input Channel = 10
Filter = 5
```

Output Shape:

```text
4 - 2 + 1 = 3
```

따라서:

```text
(None, 3, 3, 5)
```

Parameter:

```text
(2 × 2 × 10 + 1) × 5
= 205
```

---

## 11. MNIST 데이터셋

MNIST는 손으로 쓴 숫자 이미지 데이터셋이다.

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

Shape:

```python
print(x_train.shape, y_train.shape)
# (60000, 28, 28) (60000,)

print(x_test.shape, y_test.shape)
# (10000, 28, 28) (10000,)
```

의미:

```text
x_train
60000장의 흑백 이미지
각 이미지 크기 = 28 × 28

y_train
각 이미지의 정답 숫자
0 ~ 9
```

---

## 12. MNIST는 처음에는 3차원이다

중요:

```text
(60000, 28, 28)
```

은 실제로 **3차원 배열**이다.

```text
samples
height
width
```

흑백 이미지라 channel 축이 생략되어 있다.

하지만 `Conv2D`에 넣으려면 보통 channel 축을 추가해:

```text
(60000, 28, 28, 1)
```

로 만든다.

예:

```python
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
```

또는:

```python
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
```

> MNIST가 처음부터 4차원인 것이 아니라, `Conv2D` 입력을 위해 channel 차원을 추가해서 4차원으로 만든다.

---

## 13. MNIST 클래스 확인

```python
print(
    np.unique(
        y_train,
        return_counts=True
    )
)
```

MNIST에는:

```text
0 1 2 3 4 5 6 7 8 9
```

총 10개의 클래스가 있다.

따라서 MNIST는 **10개 클래스를 구분하는 다중분류 문제**이다.

---

## 14. MNIST 이미지 확인

```python
import matplotlib.pyplot as plt

plt.imshow(
    x_train[680],
    cmap='gray'
)

plt.show()
```

`x_train[680]`은 681번째 이미지를 의미한다.

---

## 15. 이미지 Pixel 값

MNIST 원본 Pixel 값:

```python
print(np.max(x_train), np.min(x_train))
```

결과:

```text
255 0
```

즉 Pixel 범위는:

```text
0 ~ 255
```

이다.

---

## 16. Image Scaling - 0 ~ 1

이미지에서는 간단하게:

```python
x_train = x_train / 255.
x_test = x_test / 255.
```

로 Scaling할 수 있다.

그러면:

```text
0 ~ 255
↓
0 ~ 1
```

로 바뀐다.

확인:

```python
print(np.max(x_train), np.min(x_train))
# 1.0 0.0
```

---

## 17. Image Scaling -1 ~ 1

원본 Pixel 값이 **0~255 상태일 때**:

```python
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
```

로 변환하면:

```text
0     → -1
127.5 → 0
255   → 1
```

즉:

```text
-1 ~ 1
```

범위가 된다.

---

## 19. Scaler 클래스를 꼭 써야 하는가?

이미지는 Pixel 값 범위가 명확해서 초반에는 직접 계산하는 방식이 편하다.

```python
x_train = x_train / 255.
```

또는:

```python
x_train = (x_train - 127.5) / 127.5
```

처럼 사용할 수 있다.

`MinMaxScaler` 같은 클래스를 사용할 수도 있지만,
이미지에서는 이런 직접 Scaling 방식이 자주 사용된다.

---

CNN에서는:

```text
Image
↓
Kernel / Filter
↓
Convolution
↓
Feature Map
↓
Edge / Line
↓
Shape
↓
고차원 특징
↓
분류
```
과정을 통해 이미지 특징을 학습한다.

