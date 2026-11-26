#23-1 copy
import numpy as np
import time
import pandas as pd
import my_util
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


#1.data
datasets = load_iris()

x = datasets.data
y = datasets['target']

# print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([50, 50, 50]))

# ##################### one hot encoding 1. to_categorical ############################
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape) #(150, 3) ->onehot 먼저하는 이유 그냥 먼저하고 잘라야 한번 자르기때문에

##################### one hot encoding 2. pandas ############################
# y = pd.get_dummies(y, dtype=int)
# print(y)

###################### one hot encoding 3. skleran ############################
from sklearn.preprocessing import OneHotEncoder #onehotencoder가져왔으니까 정의
ohe = OneHotEncoder(sparse_output=False) # sparse->혼돈행렬 형태로 나옴 그래서 sparse_output=False
y = y.reshape(-1,1)
y = ohe.fit_transform(y)#벡터형태 데이터를 행렬형태 데이터로 변환해야함 (150,)-> (150,1) reshape
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=75,
    shuffle=True,
    stratify=y,
)

# print(np.unique(y_train, return_counts=True))
# print(np.unique(y_test, return_counts=True))
# print(x_train.shape, x_test.shape)  #(120, 4) (30, 4)
# print(y_train.shape, y_test.shape)  #(120, 3) (30, 3) 변경됨 

#2.model
model = Sequential()
# model.add(Dense(60, input_dim=4, activation='relu'))
model.add(Dense(60, input_shape=(4,), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) #뉴 페이스 출력값3개 softmax가 모든값을 1이상 넘지 않게 만듬

"""
원데이터 -> input_shape
(n,4) -> input_shape (4,)
(n,100,3) - > input_shape(100,3)
(n,100,100,3) -> input_shape(100,100,3)
"""

#3.compile, train
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True,
)
start_time = time.time()
batch_size=5
history = model.fit(x_train, y_train, epochs=1000, batch_size=5, 
         verbose=1, validation_split=0.2, callbacks=[es],)
train_time = time.time() - start_time

#4.evaluate, predict
result = model.evaluate(x_test,y_test)
print('loss : ', result[0]) # 그냥 loss적으면 metrics들어가있어서 2개나옴
print('acc : ', round(result[1],2))

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_predict)
print("acc :  ", acc_score)
print("걸린시간 : ", round(train_time, 2), "초")
print(y_test)



my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 78,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = result,
    # r2 = r2
)



