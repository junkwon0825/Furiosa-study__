# Day15 — CNN Functional API & Image Data Processing

## 1. CNN 모델을 함수형 모델로 구성

기존 `Sequential()` 방식뿐 아니라 **Functional API**로 CNN을 만드는 방법을 배움.

```python
inputs = Input(shape=(100, 100, 1))

x = Conv2D(32, (3,3), activation='relu')(inputs)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
```

흐름은 동일하다.

```text
Input
 ↓
Conv2D
 ↓
Pooling
 ↓
Dense
 ↓
Output
```

`Sequential`은 순서대로 쌓는 방식이고, Functional API는 **입력과 출력의 연결 관계를 직접 정의**한다.

---

## 2. 이미지도 결국 숫자 데이터

컴퓨터가 보는 이미지는 사진 자체가 아니라 **픽셀 숫자의 배열**이다.

흑백 이미지:

```text
100 × 100 × 1
```

RGB 이미지:

```text
100 × 100 × 3
```

여러 장이면:

```text
(이미지 개수, 높이, 너비, 채널)
```

예:

```python
x_train.shape
# (160, 100, 100, 1)
```

즉 이미지도 결국 NumPy 배열 또는 Tensor 형태의 수치 데이터로 모델에 들어간다.

> 이미지 딥러닝도 결국 숫자 배열을 모델에 넣어서 계산하는 것.

단, TensorFlow 내부 연산은 주로 Tensor로 진행되고 NumPy는 데이터 확인·가공·저장 등에 매우 많이 사용된다.

---

## 3. ImageDataGenerator

이미지를 읽어오면서 동시에 전처리할 수 있다.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255
)
```

그리고:

```python
xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale'
)
```

이미지 파일이 다음 과정으로 처리된다.

```text
사진 파일
 ↓
크기 변환
 ↓
픽셀 수치화
 ↓
rescale
 ↓
batch 생성
```

---

## 4. Batch Size

예를 들어 전체 데이터가:

```text
(80, 100, 100, 1)
```

이고:

```python
batch_size = 10
```

이면 한 번에 10장씩 가져온다.

```text
전체 80장

1 batch → (10,100,100,1)
2 batch → (10,100,100,1)
3 batch → (10,100,100,1)
...
8 batch → (10,100,100,1)
```

즉:

```text
80 / 10 = 8 batches
```

한 epoch에서 모델이 8번에 나눠서 데이터를 받는다.

> `batch_size=10`은 데이터를 10개로 나눈다는 뜻이 아니라, 한 번에 이미지 10장을 사용한다는 뜻이다.

---

## 5. Iterator / DirectoryIterator

```python
xy_train = train_datagen.flow_from_directory(...)
```

이렇게 만들면 `xy_train`은 단순한 NumPy 배열이 아니라 **Iterator 객체**다.

Iterator의 역할:

```text
전체 이미지 관리
 ↓
batch 단위로 꺼냄
 ↓
x와 y를 함께 제공
 ↓
다음 batch 제공
 ↓
반복
```

예:

```python
x_batch, y_batch = xy_train[0]
```

첫 번째 batch를 가져온다.

> Iterator는 전체 이미지 데이터를 관리하면서 요청할 때마다 batch 단위로 x, y를 공급해주는 객체다.

---

## 6. Iterator 정보 확인

```python
xy_train.batch_size
```

설정한 batch size 확인.

```python
xy_train.samples
```

전체 이미지 개수 확인.

```python
len(xy_train)
```

한 epoch에서 몇 개의 batch가 있는지 확인.

```python
xy_train[0][0].shape
```

첫 번째 batch의 x shape 확인.

예:

```python
print(xy_train.samples)
# 80

print(xy_train.batch_size)
# 10

print(len(xy_train))
# 8

print(xy_train[0][0].shape)
# (10, 100, 100, 1)
```

---

## 7. `xy_train[0]` 구조

```python
xy_train[0]
```

안에는:

```python
(x_batch, y_batch)
```

두 개가 있다.

따라서:

```python
xy_train[0][0]
```

은 `x`.

```python
xy_train[0][1]
```

은 `y`.

즉:

```python
x_train = xy_train[0][0]
y_train = xy_train[0][1]
```

처럼 사용할 수 있다.

하지만 주의:

> `xy_train[0]`은 전체 train 데이터가 아니라 첫 번째 batch만 가져온다.

예:

```text
전체 데이터 = 8005장
batch_size = 5000
```

이면:

```python
xy_train[0][0]
```

은 전체 8005장이 아니라 첫 번째 5000장만 가져온다.

---

## 8. Batch Size를 너무 크게 하면 메모리 문제가 생길 수 있음

이미지는 데이터 하나 자체가 크다.

예를 들어:

```text
5000 × 100 × 100 × 3
```

이면 숫자 개수는:

```text
150,000,000개
```

`float32` 하나가 4 byte이므로 약 600MB 수준이다.

따라서:

```python
batch_size=5000
```

처럼 아주 큰 batch를 사용해 데이터를 한꺼번에 NumPy 배열로 만들면 RAM/GPU 메모리 부족이 생길 수 있다.

Iterator를 사용하는 이유 중 하나는:

```text
전체 데이터를 한꺼번에 메모리에 올리지 않고
↓
batch 단위로 조금씩 가져오기
```

이다.

---

## 9. 이진분류에서는 모델 단순화

Cat/Dog, 정상/비정상처럼 클래스가 2개라면 보통:

```python
Dense(1, activation='sigmoid')
```

를 사용한다.

Loss:

```python
loss='binary_crossentropy'
```

예측:

```python
y_pred = model.predict(x_test)
```

sigmoid 출력 예:

```text
0.08
0.91
0.76
0.13
```

0/1로 변환:

```python
y_pred = (y_pred > 0.5).astype(int)
```

결과:

```text
0.08 → 0
0.91 → 1
0.76 → 1
0.13 → 0
```

이진분류에서는 일반적으로 출력 노드를 여러 개 만들거나 one-hot encoding을 사용할 필요가 없다.

---

## 10. 좋은 파라미터를 직접 찾아보는 경험

CNN에서는 정해진 하나의 정답 모델이 없다.

직접 조절할 수 있는 주요 값:

- Conv filter 수
- kernel size
- Pooling 사용 여부
- Dropout 비율
- Dense units
- learning rate
- batch size
- epochs

예:

```text
모델 A
 ↓
성능 확인

모델 B
 ↓
성능 확인

파라미터 변경
 ↓
다시 비교
```

좋은 모델을 만들기 위해서는 **파라미터를 직접 바꾸고 성능을 비교해보는 경험**이 중요하다.

---
