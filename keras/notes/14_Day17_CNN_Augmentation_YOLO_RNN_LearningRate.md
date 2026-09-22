# Day 17 — CNN 데이터 증폭, YOLO v1, RNN 기초, Learning Rate

## 1. 오늘의 전체 흐름

오늘은 지금까지 배운 딥러닝 흐름을 다음처럼 연결해서 정리했다.

```text
DNN
↓
CNN
↓
CNN + Image Data Augmentation
↓
Object Detection — YOLO v1
↓
RNN — 시계열 데이터 처리
```

또한 NumPy 배열을 이용해서 특정 클래스만 골라 데이터 증폭하는 방법과, 학습 과정에서 중요한 `Learning Rate` 개념을 다시 정리했다.

---

## 2. DNN → CNN → CNN + 데이터 증폭

### DNN

DNN은 입력 데이터를 1차원 벡터 형태로 펼쳐서 Dense Layer에 넣는 방식이다.

```text
28 × 28 이미지
↓ Flatten
784
↓ Dense
Class
```

이미지의 공간적인 구조를 직접 활용하지 못한다는 단점이 있다.

### CNN

CNN은 이미지의 가로·세로 구조를 유지하면서 특징을 추출한다.

```text
Image
↓
Conv2D
↓
Feature Map
↓
Pooling
↓
Conv2D
↓
Feature Map
↓
Dense / GAP
↓
Classification
```

보통 얕은 Conv는 선·모서리 같은 단순 특징을, 더 깊은 Conv는 눈·코·바퀴 같은 부분 특징과 복잡한 객체 특징을 학습한다.

### CNN + 데이터 증폭

데이터 증폭(Data Augmentation)은 기존 이미지를 조금씩 변형해서 학습 데이터의 다양성을 늘리는 방법이다.

```python
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

| 옵션 | 의미 |
|---|---|
| `horizontal_flip=True` | 좌우 반전 |
| `vertical_flip=True` | 상하 반전 |
| `width_shift_range` | 좌우 이동 |
| `height_shift_range` | 위아래 이동 |
| `rotation_range` | 이미지 회전 |
| `zoom_range` | 확대/축소 |
| `shear_range` | 기울이기 |
| `fill_mode='nearest'` | 변형 후 빈 공간을 가까운 픽셀값으로 채움 |

모든 augmentation 옵션을 무조건 많이 넣는다고 좋은 것은 아니다. 이미지의 의미가 유지되는 범위에서 적용해야 한다.

---

## 3. MNIST / Fashion-MNIST / CIFAR 데이터 증폭

### MNIST / Fashion-MNIST

원본은 보통:

```text
(N, 28, 28)
```

CNN에 넣을 때는 채널 차원을 추가한다.

```python
x_train = x_train.reshape(-1, 28, 28, 1)
```

최종 형태:

```text
(N, 28, 28, 1)
```

흑백 이미지이므로 채널 수는 `1`이다.

### CIFAR-10 / CIFAR-100

원본부터 RGB 이미지이므로:

```text
(N, 32, 32, 3)
```

형태다. 마지막 `3`은 RGB 3개 채널이다.

---

## 4. Matrix와 Tensor

이미지 한 장은 행렬(Matrix)로 생각할 수 있다.

```text
흑백 이미지: 28 × 28
RGB 이미지: 32 × 32 × 3
```

여러 장이 모이면 Tensor 형태가 된다.

```text
Fashion-MNIST: (60000, 28, 28, 1)
CIFAR-10:      (50000, 32, 32, 3)
```

개념적으로는:

```text
Scalar
↓
Vector
↓
Matrix
↓
Tensor
```

처럼 차원이 늘어난다.

---

## 5. Python 자료구조 간단 정리

### List

```python
a = [1, 2, 3]
```

순서가 있고 수정 가능하며 중복을 허용한다.

### Tuple

```python
a = (1, 2, 3)
```

순서가 있지만 수정할 수 없다. 여러 값을 한 번에 반환할 때 자주 사용한다.

예:

```python
x_batch, y_batch = datagen.flow(...).next()
```

`next()`가 `(x_batch, y_batch)` 튜플을 반환한다.

### Dictionary

```python
a = {
    'men': 0,
    'women': 1
}
```

Key와 Value 형태로 데이터를 저장한다.

예:

```python
print(xy_data.class_indices)
```

결과:

```python
{'men': 0, 'women': 1}
```

### NumPy Vector

```python
a = np.array([1, 2, 3])
```

shape:

```text
(3,)
```

NumPy 배열에서는 Boolean Masking, Fancy Indexing 등을 사용할 수 있다.

---

## 6. Boolean Masking

여자 클래스가 `1`이라고 가정한다.

```python
women_mask = (y_train == 1)
```

예:

```python
y_train = np.array([0, 1, 0, 1, 1])
```

이면:

```text
women_mask
[False, True, False, True, True]
```

이 mask를 이용하면 여자 데이터만 추출할 수 있다.

```python
x_women = x_train[women_mask]
y_women = y_train[women_mask]
```

`women_idx`라는 이름을 써도 되지만 실제 값이 True/False 배열이라면 `women_mask`가 더 정확한 이름이다.


### 데이터 로드

```python
path = './_data/image/men_women/faces/'
save_path = './_save/keras47/'
np_path = './_data/kaggle_cat_dog_npy/'
```

이미지를 읽으면서 0~1로 스케일링한다.

```python
datagen = ImageDataGenerator(
    rescale=1./255
)
```

증폭용 generator는 별도로 만든다.

```python
datagen_aug = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,
    fill_mode='nearest'
)
```

`datagen_aug`에 다시 `rescale=1./255`를 넣지 않는 이유는 처음 `flow_from_directory()`에서 이미 0~1로 변환했기 때문이다.

```text
0~255
↓ /255
0~1
```

여기서 또 `/255`를 적용하면 증폭 데이터만 너무 작은 값이 된다.

---

## 8. `flow_from_directory()`에서 batch 꺼내기

```python
xy_data = datagen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=3000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)
```

예를 들어:

```text
Found 27167 images belonging to 2 classes.
```

라고 떠도:

```python
x, y = xy_data[0]
```

은 전체 27,167장이 아니라 **첫 번째 batch만** 가져온다.

`batch_size=3000`이면 대략:

```text
x.shape = (3000, 100, 100, 3)
y.shape = (3000,)
```

이다.

---

## 9. Train / Test 분리

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y`는 남녀 클래스 비율을 유지하면서 train/test를 분리한다.

---

## 10. 여자 데이터만 선택

여자가 `1`인지 먼저 확인한다.

```python
print(xy_data.class_indices)
```

예:

```python
{'men': 0, 'women': 1}
```

그 다음 여자만 추출한다.

```python
women_mask = (y_train == 1)

x_women = x_train[women_mask]
y_women = y_train[women_mask]
```

---

## 11. 여자 이미지 800장 선택

```python
augment_size = 800
```

여자 데이터 중 800장을 랜덤 선택한다.

```python
randidx = np.random.choice(
    x_women.shape[0],
    size=augment_size,
    replace=False
)
```

같은 index를 x와 y에 적용한다.

```python
x_w_aug = x_women[randidx].copy()
y_w_aug = y_women[randidx].copy()
```

`replace=False`는 중복 없이 뽑는다는 뜻이다. 따라서:

```text
augment_size <= 여자 원본 개수
```

여야 한다. 원본 여자 이미지보다 더 많이 뽑아야 한다면 `replace=True`가 필요하다.

---

## 12. 여자 데이터 증폭

```python
x_augmented, y_augmented = datagen_aug.flow(
    x_w_aug,
    y_w_aug,
    batch_size=augment_size,
    shuffle=False
).next()
```

흐름:

```text
flow()
↓
Iterator 생성
↓
.next()
↓
(x_batch, y_batch)
```

정리:

```text
.next()     → x, y 둘 다
.next()[0]  → x만
.next()[1]  → y만
```

따라서 다음은 가능하다.

```python
x_augmented = datagen_aug.flow(...).next()[0]
```

하지만 다음처럼 쓰면 안 된다.

```python
x_augmented, y_augmented = datagen_aug.flow(...).next()[0]
```

`[0]`을 붙이는 순간 x 하나만 남기 때문이다.

---

## 13. 기존 데이터와 증폭 데이터 합치기

```python
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
```

최종 흐름:

```text
기존 train
남자 + 여자
        +
증폭된 여자
        ↓
새로운 x_train / y_train
```

클래스별 데이터 개수는:

```python
print(np.unique(y_train, return_counts=True))
```

로 확인한다.

---

## 14. 여자만 증폭하는 전체 코드

```python
path = './_data/image/men_women/faces/'

# 이미지 로딩 + scaling
datagen = ImageDataGenerator(
    rescale=1./255
)

# augmentation 전용
datagen_aug = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,
    fill_mode='nearest'
)

xy_data = datagen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=3000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

print(xy_data.class_indices)

x, y = xy_data[0]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 여자만 추출
women_mask = (y_train == 1)

x_women = x_train[women_mask]
y_women = y_train[women_mask]

# 여자 데이터 800장 선택
augment_size = 800

randidx = np.random.choice(
    x_women.shape[0],
    size=augment_size,
    replace=False
)

x_w_aug = x_women[randidx].copy()
y_w_aug = y_women[randidx].copy()

# augmentation
x_augmented, y_augmented = datagen_aug.flow(
    x_w_aug,
    y_w_aug,
    batch_size=augment_size,
    shuffle=False
).next()

# 기존 학습 데이터에 추가
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))
```

---

## 15. YOLO v1

YOLO는 Object Detection 모델이다.

일반 CNN Classification은:

```text
사진
↓
CNN
↓
이 사진은 무엇인가?
```

를 예측한다.

YOLO는:

```text
사진
↓
CNN
↓
어디에 있는가?
+
무엇인가?
+
객체가 실제로 있는가?
```

를 한 번에 예측한다.

YOLO는 `You Only Look Once`의 약자다.

---

## 16. YOLO v1 Grid 구조

YOLO v1은 이미지를:

```text
7 × 7
```

grid로 나눈다.

```text
S = 7
S² = 49
```

객체의 중심점이 들어있는 grid cell이 해당 객체를 담당한다.

각 grid cell은 bounding box `B=2`개를 예측한다.

box 하나가 예측하는 값:

```text
x
y
w
h
confidence
```

즉 5개다.

VOC 기준 클래스가 20개라면:

```text
2 × 5 + 20 = 30
```

그래서 전체 출력은:

```text
7 × 7 × 30
```

이다.

---

## 17. YOLO 수학적 출력

이미지 입력을 `X`, 네트워크 파라미터를 `θ`라고 하면:

\[
f_\theta(X)
\]

YOLO는 다음을 예측한다.

\[
(x, y, w, h, C, P(c))
\]

```text
x, y
→ bounding box 중심

w, h
→ bounding box 크기

C
→ confidence

P(c)
→ class probability
```

YOLO v1의 큰 Loss 구조:

\[
Loss = Localization + Confidence + Classification
\]

세부적으로는:

```text
1. x, y 위치 오차
2. w, h 크기 오차
3. object confidence 오차
4. no-object confidence 오차
5. classification 오차
```

### Bounding Box 위치

\[
(x-\hat{x})^2 + (y-\hat{y})^2
\]

실제 중심과 예측 중심의 차이를 줄인다.

### Bounding Box 크기

\[
(\sqrt{w}-\sqrt{\hat{w}})^2
+
(\sqrt{h}-\sqrt{\hat{h}})^2
\]

작은 bounding box의 크기 오차도 중요하게 반영하기 위해 square root를 사용한다.

### Confidence

YOLO v1 confidence는 개념적으로:

\[
P(Object) \times IoU
\]

이다.

IoU:

\[
IoU = \frac{Prediction \cap GroundTruth}{Prediction \cup GroundTruth}
\]

예측 box와 정답 box가 얼마나 겹치는지를 나타낸다.

---

## 18. YOLO v1 — Darknet Architecture

YOLO v1은 CNN 기반 detection architecture다.

```text
Input Image
↓
Convolution
↓
Pooling
↓
Convolution
↓
Pooling
↓
...
↓
Feature Extraction
↓
Fully Connected
↓
7 × 7 × 30
```

YOLO v1은 24개의 convolution layer와 2개의 fully connected layer를 사용한다.

CNN 부분은 이미지 특징을 추출하고, 마지막 detection 부분은:

```text
Bounding Box
+
Confidence
+
Class
```

를 예측한다.

후기 YOLO 모델들은 구조가 많이 발전했지만, YOLO v1의 핵심은 **한 네트워크에서 위치와 클래스를 동시에 예측한다는 것**이다.

---

## 19. RNN 기초

RNN은 순서가 중요한 데이터를 처리하기 위해 사용한다.

대표적인 데이터:

```text
주가
날씨
센서값
음성
문장
시계열
```

CNN이 공간적인 관계를 학습한다면:

```text
CNN → 공간 관계
RNN → 시간 / 순서 관계
```

를 학습한다고 볼 수 있다.

현재 입력을 `x_t`, 이전 hidden state를 `h_(t-1)`이라고 하면:

\[
h_t = f(W_xx_t + W_hh_{t-1} + b)
\]

즉 이전 시점의 정보를 현재 계산에 같이 사용한다.

```text
x1 → h1
      ↓
x2 → h2
      ↓
x3 → h3
      ↓
x4 → h4
```

---

## 20. Learning Rate

Learning Rate는:

> 가중치를 한 번 업데이트할 때 얼마나 크게 움직일지를 정하는 값

이다.

경사하강법의 기본식:

\[
w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}
\]

여기서:

```text
w
→ weight

L
→ loss

∂L/∂w
→ gradient

η
→ learning rate
```

Learning Rate가 너무 크면 최적점을 지나칠 수 있고, 너무 작으면 학습이 매우 느려질 수 있다.

---

## 21. Optimizer와 Learning Rate

Optimizer마다 기본 learning rate가 다를 수 있다.

대표적으로:

```text
SGD      → 약 0.01
Adam     → 약 0.001
RMSprop  → 약 0.001
```

직접 설정할 수도 있다.

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

같은 learning rate라도 optimizer마다 gradient를 사용하는 방식이 다르기 때문에 단순히 숫자만으로 비교하면 안 된다.

---

## 22. Forward → Loss → Backpropagation → Update

딥러닝 학습의 핵심 흐름:

```text
Input
↓
Forward Propagation
↓
Prediction
↓
Loss 계산
↓
Backpropagation
↓
Gradient 계산
↓
Optimizer
↓
Weight Update
↓
다시 Forward
```

## 23. Loss와 Weight의 관계

간단한 선형 모델 + MSE에서는 하나의 weight에 대해 loss가 2차 함수 형태로 나타날 수 있다.

예:

\[
L(w)=(wx-y)^2
\]

전개하면:

\[
L(w)=x^2w^2-2xyw+y^2
\]

즉 `w`에 대한 2차 함수다.

하지만 실제 CNN, RNN, Transformer처럼 파라미터가 매우 많은 딥러닝 모델의 loss landscape는 단순한 포물선 하나가 아니다.

```text
수십만 ~ 수십억 개 weight
↓
고차원 Loss Landscape
↓
복잡한 비선형 최적화 문제
```

따라서 **단순 예제에서는 Loss-Weight 관계를 2차 함수로 이해할 수 있지만, 실제 딥러닝에서는 훨씬 복잡한 고차원 함수**라고 보는 것이 맞다.

---
