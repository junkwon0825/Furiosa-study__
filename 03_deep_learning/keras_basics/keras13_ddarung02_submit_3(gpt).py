import numpy as np
import pandas as pd
import time
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

import my_util


# =========================================================
# 0. random seed
# =========================================================
np.random.seed(42)
tf.random.set_seed(42)


# =========================================================
# 1. DATA
# =========================================================
path = "./_data/ddarung/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "submission.csv", index_col=0)

print("train :", train_csv.shape)
print("test :", test_csv.shape)

print("\ntrain 결측치")
print(train_csv.isnull().sum())

print("\ntest 결측치")
print(test_csv.isnull().sum())


# =========================================================
# 2. X / Y 분리
# =========================================================
x = train_csv.drop(["count"], axis=1)
y = train_csv["count"]


# =========================================================
# 3. TRAIN / VALIDATION SPLIT
# =========================================================
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    shuffle=True,
    random_state=79
)


# =========================================================
# 4. 결측치 처리
#    train 데이터 기준 중앙값으로 채움
# =========================================================
imputer = SimpleImputer(strategy="median")

x_train = imputer.fit_transform(x_train)

# train에서 구한 중앙값을 사용
x_test = imputer.transform(x_test)
test_csv_processed = imputer.transform(test_csv)


# =========================================================
# 5. Scaling
# =========================================================
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

# train 기준으로 scaling
x_test = scaler.transform(x_test)
test_csv_processed = scaler.transform(test_csv_processed)


# =========================================================
# 6. MODEL
# =========================================================
model = Sequential()

model.add(Dense(128, input_dim=9, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))

# 회귀이므로 마지막은 activation 없음
model.add(Dense(1))


# =========================================================
# 7. COMPILE
# =========================================================
model.compile(
    loss="mse",
    optimizer=Adam(learning_rate=0.001)
)


# =========================================================
# 8. CALLBACK
# =========================================================

# validation loss가 30 epoch 동안 좋아지지 않으면 종료
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=30,
    restore_best_weights=True
)

# validation loss가 개선되지 않으면 learning rate 감소
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    patience=10,
    factor=0.5,
    min_lr=1e-6
)


# =========================================================
# 9. TRAIN
# =========================================================
start_time = time.time()

batch_size = 16

history = model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=batch_size,

    # x_train 중 20%를 validation으로 사용
    validation_split=0.2,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)

train_time = time.time() - start_time


# =========================================================
# 10. EVALUATE
# =========================================================
loss = model.evaluate(x_test, y_test)

print("\n============================")
print("Test MSE :", loss)


# =========================================================
# 11. VALIDATION PREDICTION
# =========================================================
y_predict = model.predict(x_test).flatten()

r2 = r2_score(y_test, y_predict)

rmse = np.sqrt(
    mean_squared_error(y_test, y_predict)
)

print("R2 :", r2)
print("RMSE :", rmse)


# =========================================================
# 12. DACON TEST PREDICTION
# =========================================================
y_submit = model.predict(
    test_csv_processed
).flatten()


# 따릉이 대여량은 음수가 될 수 없음
y_submit = np.clip(
    y_submit,
    a_min=0,
    a_max=None
)


# =========================================================
# 13. SUBMISSION
# =========================================================
submission["count"] = y_submit

print("\nSubmission")
print(submission.head())

print("\n결측치 확인")
print(submission.isnull().sum())


submission.to_csv(
    path + "submission_improved.csv",
    index=True
)

print(
    "\nsubmission_improved.csv 저장 완료"
)


# =========================================================
# 14. RECORD
# =========================================================
my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=79,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    test_loss=loss,
    r2=r2
)