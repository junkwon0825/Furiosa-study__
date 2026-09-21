# Day16 — Image Classification, Model Saving, Data Augmentation & ML Transition


## 2. 세상에 완벽한 모델은 없다

머신러닝/딥러닝 모델은 항상 데이터에 따라 성능이 달라진다.

```text
모델 구조
+
데이터 품질
+
전처리
+
파라미터
+
학습 방법
```

이 조합에 따라 결과가 달라진다.

따라서:

```text
"이 모델이 무조건 최고"
```

보다는

```text
"이 데이터에서는 이 설정이 잘 나왔다"
```

라고 보는 것이 더 정확하다.

여러 모델과 파라미터를 직접 실험하면서 성능을 비교하는 경험이 중요하다.

---

## 3. 이미지 분류 복습

오늘까지 직접 만들어본 이미지 분류 예제:

```text
Horse / Human
Rock / Paper / Scissors
Men / Women
Cat / Dog
```

### 이진분류

예:

```text
Men / Women
Cat / Dog
Horse / Human
```

출력층:

```python
model.add(Dense(1, activation='sigmoid'))
```

Loss:

```python
loss='binary_crossentropy'
```

예측:

```python
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
```

또는:

```python
y_predict = (y_predict > 0.5).astype(int)
```

### 다중분류

예:

```text
Rock / Paper / Scissors
```

클래스가 3개라면:

```python
model.add(Dense(3, activation='softmax'))
```

정수 라벨을 그대로 사용하면:

```python
loss='sparse_categorical_crossentropy'
```

예측:

```python
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
```

---

## 4. Train/Test 폴더가 따로 없을 때

이미지가 클래스별 폴더로 라벨링되어 있지만 train/test 폴더가 따로 나뉘어 있지 않다면 전체 데이터를 불러온 뒤 `train_test_split()`을 사용할 수 있다.

예:

```text
men_women/
├── men/
└── women/
```

```python
xy_data = datagen.flow_from_directory(
    path,
    target_size=(150,150),
    batch_size=2000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

x, y = xy_data[0]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y`를 사용하면 클래스 비율을 최대한 유지하면서 train/test를 나눈다.

주의:

```python
xy_data[0]
```

은 **첫 번째 batch만 가져온다.**

전체 데이터가 27,000장인데:

```python
batch_size=2000
```

이면 첫 번째 2,000장만 가져오는 것이다.

---

## 5. Image → NumPy 변환

이미지를 직접 모델에 넣기 위해 Pillow 이미지 객체를 NumPy 배열로 변환할 수 있다.

```python
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
```

이미지 불러오기:

```python
img = load_img(
    path + '1.jpg',
    target_size=(100,100)
)
```

타입:

```text
<class 'PIL.Image.Image'>
```

NumPy 배열로 변환:

```python
arr = img_to_array(img)
```

shape:

```text
(100, 100, 3)
```

타입:

```text
<class 'numpy.ndarray'>
```

---

## 6. Batch 차원 추가

CNN은 일반적으로 batch 차원을 포함한 4차원 입력을 사용한다.

현재:

```text
(100, 100, 3)
```

이면 이미지 한 장의 shape이다.

```python
arr = np.expand_dims(arr, axis=0)
```

결과:

```text
(1, 100, 100, 3)
```

의미:

```text
1   = 이미지 개수
100 = height
100 = width
3   = RGB channel
```

---

## 7. Rescale

일반적인 8bit 이미지의 픽셀 범위:

```text
0 ~ 255
```

따라서:

```python
rescale=1./255
```

를 사용하면:

```text
0 ~ 255
↓
0 ~ 1
```

범위로 변환된다.

학습할 때 `rescale=1./255`를 사용했다면 새로운 이미지를 `predict()`할 때도 같은 전처리를 적용해야 한다.

단, 이미 0~1 범위인 데이터를 또 `/255`하면 안 된다.

확인:

```python
print(arr.min())
print(arr.max())
```

---

## 8. Men / Women 이미지 분류 모델

오늘까지 사용한 모델 예제:

```python
import numpy as np
import time

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.layers import Flatten, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. Data
path = './_data/image/men_women/'
save_path = './_save/keras47/'
np_path = './_data/kaggle_cat_dog_npy/'

x_train = np.load(np_path + 'keras47_01_x_train.npy')
y_train = np.load(np_path + 'keras47_01_y_train.npy')
x_test = np.load(np_path + 'keras47_01_x_test.npy')
y_test = np.load(np_path + 'keras47_01_y_test.npy')

# 2. Model
model = Sequential()

model.add(
    Conv2D(
        16,
        (5,5),
        input_shape=(150,150,3),
        activation='relu',
        padding='same',
        strides=1
    )
)

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(
    Conv2D(
        64,
        (5,5),
        activation='relu',
        padding='same'
    )
)

model.add(
    Conv2D(
        32,
        (5,5),
        activation='relu',
        padding='same'
    )
)

model.add(
    Conv2D(
        16,
        (5,5),
        activation='relu',
        padding='same'
    )
)

model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D())

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# 저장된 가중치 불러오기
model.load_weights(
    save_path + 'final.weights.h5'
)

# 3. Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
```

---

## 9. Weight 저장 복습

### `save_weights()`

모델의 학습된:

```text
Weight
Bias
```

만 저장한다.

```python
model.save_weights(
    save_path + 'final.weights.h5'
)
```

불러올 때:

```python
model.load_weights(
    save_path + 'final.weights.h5'
)
```

중요:

> `load_weights()`를 사용할 때는 저장 당시와 모델 구조가 동일해야 한다.

예를 들어 저장 모델:

```text
6 layers
```

현재 모델:

```text
10 layers
```

이면 weight shape가 맞지 않아 오류가 발생할 수 있다.

---

## 10. ModelCheckpoint 복습

`ModelCheckpoint`는 학습하면서 가장 좋은 모델 또는 가중치를 자동으로 저장해준다.

```python
from tensorflow.keras.callbacks import ModelCheckpoint
```

가중치만 저장:

```python
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=True,
    filepath=save_path + 'best.weights.h5',
    verbose=1
)
```

학습:

```python
model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=16,
    validation_split=0.2,
    callbacks=[es, mcp]
)
```

흐름:

```text
Epoch 진행
↓
val_loss 확인
↓
이전보다 좋아짐?
↓
YES
↓
best.weights.h5 저장
```

---

## 11. EarlyStopping + ModelCheckpoint

둘은 역할이 다르다.

### EarlyStopping

```python
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
    verbose=1
)
```

역할:

```text
val_loss 개선이 일정 기간 없음
↓
학습 종료
↓
best weight 복구
```

### ModelCheckpoint

```text
best weight를 실제 파일로 저장
```

따라서 같이 사용할 수 있다.

```python
callbacks=[es, mcp]
```

---


## 13. 새로운 이미지 Predict

이미 저장한 개인 사진을 불러와 예측할 수 있다.

```python
arr = np.load(
    np_path + "keras48_oo.npy"
)
```

학습 데이터가 0~1 범위였고 현재 이미지가 0~255라면:

```python
arr = arr / 255.
```

예측:

```python
y_predict = model.predict(arr)[0][0]

print("예측값 :", y_predict)

if y_predict < 0.5:
    print("남자")
else:
    print("여자")
```

주의:

클래스 번호는 데이터 생성 시:

```python
print(xy_data.class_indices)
```

로 반드시 확인한다.

예:

```text
{'men': 0, 'women': 1}
```

---

## 14. Accuracy와 개별 Predict 차이

### `accuracy_score`

```python
accuracy_score(
    y_test,
    y_predict
)
```

전체 test 데이터에서:

```text
실제 정답
vs
모델 예측
```

을 비교해 정확도를 계산한다.

### 개인 사진 Predict

개인 이미지 한 장의 클래스가 무엇인지 확인할 때는:

```python
model.predict(arr)
```

결과를 확인한다.

즉:

```text
accuracy_score
→ 모델 전체 평가

model.predict
→ 개별 데이터 예측
```

---

## 15. Data Augmentation

데이터 증폭은 기존 이미지를 조금씩 변형해 새로운 학습 이미지를 만드는 기법이다.

사용 가능한 예:

```python
datagen = ImageDataGenerator(
    rescale=1./255,

    horizontal_flip=True,
    vertical_flip=False,

    width_shift_range=0.1,
    height_shift_range=0.1,

    rotation_range=15,

    zoom_range=0.1,

    shear_range=0.1,

    fill_mode='nearest'
)
```

하지만 모든 옵션을 무조건 적용하면 사진이 지나치게 변형될 수 있다.

> 데이터의 의미를 유지하는 범위에서 증폭해야 한다.

---

## 16. `datagen.flow()`

NumPy 배열을 ImageDataGenerator에 직접 넣을 수 있다.

```python
it = datagen.flow(
    arr,
    batch_size=1
)
```

결과:

```text
NumpyArrayIterator
```

한 batch 가져오기:

```python
batch = next(it)
```

shape:

```text
(1, 100, 100, 3)
```

---

## 17. 데이터 증폭 시각화

```python
fig, ax = plt.subplots(
    nrows=1,
    ncols=5,
    figsize=(5,5)
)

for i in range(5):

    batch = next(it)

    batch = batch.reshape(
        100,
        100,
        3
    )

    ax[i].imshow(batch)

    ax[i].axis('off')

plt.show()
```

흐름:

```text
원본 이미지 1장
↓
ImageDataGenerator
↓
매번 조금씩 다른 이미지
↓
5개 생성
↓
subplot으로 확인
```

---

## 18. `plot()`과 `subplot()`

### `plot()`

하나의 그래프를 그릴 때 사용.

```python
plt.plot(x, y)
plt.show()
```

예:

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
```

### `subplots()`

여러 그래프 또는 이미지를 한 화면에 배치할 때 사용한다.

```python
fig, ax = plt.subplots(
    nrows=1,
    ncols=5
)
```

뜻:

```text
1행 × 5열
```

그리고:

```python
ax[0]
ax[1]
ax[2]
...
```

각 위치에 이미지 또는 그래프를 넣는다.

---

## 19. 데이터 분석 3대장

보통 데이터 분석의 기본 도구로:

```text
NumPy
Pandas
Matplotlib
```

을 많이 사용한다.

### NumPy

```text
배열
shape
reshape
axis
행렬 연산
```

### Pandas

```text
CSV
DataFrame
컬럼
결측치
데이터 전처리
```

### Matplotlib

```text
데이터 시각화
학습 결과
loss
accuracy
```

---

## 20. 전통적인 머신러닝도 여전히 중요

딥러닝과 LLM이 발전했다고 해서 전통적인 머신러닝이 사라진 것은 아니다.

특히 **정형 데이터(Tabular / Matrix Data)**에서는 Tree 기반 모델이 매우 강력하다.

대표적으로:

```text
Decision Tree
Random Forest
Boosting
```

그리고 실전에서 많이 사용하는 Boosting 계열:

```text
XGBoost
LightGBM
CatBoost
```

---

## 21. 정형 데이터 3대장

### XGBoost

```text
XGBoost
= eXtreme Gradient Boosting
```

강력한 Gradient Boosting 기반 모델.

### LightGBM

```text
Light Gradient Boosting Machine
```

빠른 학습 속도와 큰 데이터 처리에 강점이 있다.

### CatBoost

```text
Categorical Boosting
```

범주형 데이터 처리에 강점이 있다.

정형 데이터에서는:

```text
XGBoost
LightGBM
CatBoost
```

세 모델을 비교해보는 것이 좋은 baseline이 될 수 있다.

또한 이후에는:

```text
GridSearch
RandomizedSearch
Bayesian Optimization
```

같은 방법으로 파라미터 탐색을 자동화할 수 있다.

---

## 22. Scikit-learn → Tensor → LLM

앞으로 이어질 흐름:

```text
Scikit-learn
↓
전통적인 머신러닝
↓
데이터 전처리 / 평가 / 모델 선택
↓
Tensor
↓
딥러닝 계산 구조
↓
PyTorch
↓
Transformer
↓
LLM
```

---

## 23. Tensor 기초에서 중요하게 볼 것

Tensor는 딥러닝에서 사용하는 다차원 숫자 배열이다.

앞으로 중요하게 볼 개념:

```text
shape
dtype
rank
axis

reshape
transpose

broadcasting

matrix multiplication
```

예:

```python
tf.constant()
tf.Variable()

tf.reshape()
tf.transpose()

tf.matmul()

tf.reduce_mean()
tf.reduce_sum()
```

LLM에서도 Tensor shape와 행렬곱이 계속 등장한다.

---

