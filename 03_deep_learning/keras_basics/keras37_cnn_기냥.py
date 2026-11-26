from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1))) # 필터 크기
model.add(Conv2D(5, (2,2))) # 필터 크기

model.summary()
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 4, 4, 10)            │              50 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 3, 3, 5)             │             205 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 255 (1020.00 B)
#  Trainable params: 255 (1020.00 B)
#  Non-trainable params: 0 (0.00 B)







# 36-2 copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Dropout,
    Flatten,
    MaxPooling2D
)

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


# ============================================================
# 1. DATA
# ============================================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train))   # 255 0
print(np.max(x_test), np.min(x_test))     # 255 0


# ============================================================
# Scaling : -1 ~ 1
# ============================================================

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

print(np.max(x_train), np.min(x_train))   # 1.0 -1.0
print(np.max(x_test), np.min(x_test))     # 1.0 -1.0


# ============================================================
# CNN 입력을 위해 3차원 -> 4차원
# (batch, height, width, channel)
# ============================================================

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(x_train.shape)   # (60000, 28, 28, 1)
print(x_test.shape)    # (10000, 28, 28, 1)


# ============================================================
# y OneHotEncoding
# ============================================================

ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)

# train에서 학습한 OneHotEncoder 그대로 사용
y_test = ohe.transform(y_test)

print(y_train.shape)   # (60000, 10)
print(y_test.shape)    # (10000, 10)


# ============================================================
# 2. MODEL
# ============================================================

model = Sequential()


# ------------------------------------------------------------
# 28 x 28 x 1
#
# padding='same'
# -> Conv를 해도 가로/세로 크기 유지
#
# strides=1
# -> kernel이 한 칸씩 이동
# ------------------------------------------------------------

model.add(
    Conv2D(
        64,
        (3, 3),
        input_shape=(28, 28, 1),
        activation='relu',
        padding='same',
        strides=1
    )
)

# (28, 28, 64)


model.add(
    Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same',
        strides=1
    )
)

# (28, 28, 64)


# ------------------------------------------------------------
# MaxPooling
#
# 2 x 2 영역에서 가장 큰 값만 남김
#
# 28 x 28 -> 14 x 14
# ------------------------------------------------------------

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

# (14, 14, 64)

model.add(Dropout(0.2))


# ------------------------------------------------------------
# 특징 채널 증가
# ------------------------------------------------------------

model.add(
    Conv2D(
        128,
        (3, 3),
        activation='relu',
        padding='same',
        strides=1
    )
)

# (14, 14, 128)


# ------------------------------------------------------------
# stride = 2
#
# kernel이 두 칸씩 이동
#
# padding='same'이지만
# stride=2라서 크기는 절반 정도 감소
#
# 14 x 14 -> 7 x 7
# ------------------------------------------------------------

model.add(
    Conv2D(
        128,
        (3, 3),
        activation='relu',
        padding='same',
        strides=2
    )
)

# (7, 7, 128)

model.add(Dropout(0.3))


# ------------------------------------------------------------
# 마지막 특징 추출
# ------------------------------------------------------------

model.add(
    Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same',
        strides=1
    )
)

# (7, 7, 64)


# ============================================================
# CNN -> DNN
# ============================================================

model.add(Flatten())

# 7 * 7 * 64
# = 3136


model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(Dropout(0.3))


model.add(
    Dense(
        64,
        activation='relu'
    )
)


# MNIST : 10 classes
model.add(
    Dense(
        10,
        activation='softmax'
    )
)


model.summary()


# ============================================================
# 3. COMPILE
# ============================================================

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)


# ============================================================
# EarlyStopping
# ============================================================

es = EarlyStopping(
    monitor='val_loss',          # validation loss 감시
    mode='min',                  # val_loss는 작을수록 좋음
    patience=10,                 # 10 epoch 동안 개선 없으면 중단
    restore_best_weights=True,   # 가장 좋았던 weight로 복구
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

start_time = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=120,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)

end_time = time.time()


# ============================================================
# 4. EVALUATE
# ============================================================

print('=============== model.evaluate ===============')

loss = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print('loss : ', loss[0])
print('acc : ', loss[1])


# ============================================================
# PREDICT
# ============================================================

y_predict = model.predict(x_test)

# softmax 확률 -> 가장 높은 클래스 번호
y_predict = np.argmax(
    y_predict,
    axis=1
).reshape(-1, 1)


# one-hot -> 실제 클래스 번호
y_test_argmax = np.argmax(
    y_test,
    axis=1
).reshape(-1, 1)


acc_score = accuracy_score(
    y_test_argmax,
    y_predict
)

print('accuracy_score : ', acc_score)

print(
    '걸린시간 : ',
    round(end_time - start_time, 2),
    '초'
)
