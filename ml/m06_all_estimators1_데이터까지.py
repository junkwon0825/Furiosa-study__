import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.utils import all_estimators    
import sklearn as sk
import time
import warnings
warnings.filterwarnings('ignore')

#1. data
iris = load_iris(return_X_y=True)
print(iris)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digits]
data_name = ['iris', 'cancer', 'wine', 'digits']


#2. model
all = all_estimators(type_filter='classifier')
# all = all_estimators(type_filter='regressor')
n_splits = 5
kfold = StratifiedKFold(
    n_splits=n_splits, shuffle=True, random_state=333)

start = time.time()
for index, value in enumerate(datasets):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size=0.8, random_state=123, stratify=y #회기에서 쓸 수 없음 
    )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("=================", data_name[index], "============")
    model_name=[]
    model_acc=[]

    for name, model in all:
        #print(model)
        try:
            model = model()

            #3. train
            # model.fit(x_train, y_train)
            scores = cross_val_score(model, x_train, y_train, cv=kfold)

            #4. 평가
            y_pred = cross_val_predict(model,x_test,y_test,cv=kfold)

            acc = np.round(accuracy_score(y_test, y_pred),4)
            model_name.append(name)
            model_acc.append(acc)
            
        except:
            continue
            # print(name, '모델 오류')    

    # model_name = np.array(model_name)
    model_acc = np.array(model_acc)
    max_index = np.argmax(model_acc)
    print('최고의 모델 :', model_name[max_index], 'ACC:', model_acc[max_index])

end = time.time()
print("걸린시간 :", round(end - start,2), '초')