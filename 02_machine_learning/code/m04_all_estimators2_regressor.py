import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.utils import all_estimators    
import sklearn as sk

import warnings
warnings.filterwarnings('ignore')

#1. data
x, y = load_iris(return_X_y=True)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size=0.8, random_state=123, stratify=y #회기에서 쓸 수 없음 
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. model
# all = all_estimators(type_filter='classifier')
all = all_estimators(type_filter='regressor')

print('all models :', all)
print('sk version : ', sk.__version__)
print('모델의 개수 :', len(all))
# sk version :  1.9.1
# 모델의 개수 : 44 ,모델의 개수 : 55

for name, model in all:
    #print(model)
    try:
        model = model()

        #3. train
        model.fit(x_train, y_train)

        #4. 평가
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 : ', acc)
    except:
        # continue
        print(name, '모델 오류')    
