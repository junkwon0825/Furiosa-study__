import numpy as np
import time
import pandas as pd
import my_util
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


#1. data
path = "./_data/kaggle_santander/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sample_submission.csv", index_col=0)

x = train_csv.drop(['target'], axis=1) 
print(x.shape) 
y = train_csv['target']

print(y.shape) #(200000,)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(sparse_output=False) 
# y = np.array(y) 
y= y.to_numpy()
y = y.reshape(-1,1)
print(y.shape) #(200000, 1)
y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=90,
    stratify=y,
)

print(x_train.shape, x_test.shape)  #(160000, 200) (40000, 200)
print(y_train.shape, y_test.shape) #(160000, 2) (40000, 2)

#2. model
model = Sequential()
model.add(Dense(800, input_dim=200, activation='relu'))
model.add(Dense(600, activation='relu')) 
model.add(Dense(400)) 
model.add(Dense(200, activation='relu'))
model.add(Dense(100))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax')) 


#3. compile , train
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'], 
              ) # 이진분류 loss도 저것만 씀 분류모델
es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    patience=300,
    restore_best_weights=True, 
)

start_time = time.time()
batch_size=500
history = model.fit(x_train, y_train, epochs=1000, batch_size = 500,
                    verbose=1, validation_split=0.2,
                    callbacks=[es],
                    )

train_time = time.time() - start_time


#4. evaluate, predict
loss = model.evaluate(x_test,y_test)
print("=====================================")
print("loss : ", round(loss[0],4))
print("acc :  ", round(loss[1],4))

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_predict)
print("acc :  ", acc_score)
print("걸린시간 : ", round(train_time, 2), "초")

r2 = r2_score(y_test, y_predict)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print(f"RMSE : {rmse : .2f}")


y_submit = model.predict(test_csv)
submission['target'] = y_submit

submission.to_csv(path + "submit/" + "submit_0910_1030.csv", index=True)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 78,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2 = r2
)


# acc :   0.9095
# 걸린시간 :  810.6 초
# RMSE :  0.30




