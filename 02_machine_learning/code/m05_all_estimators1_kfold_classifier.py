import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings('ignore')


#1. data
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle=True,
    train_size=0.8,
    random_state=123,
    stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=333
)


#2. model
all = all_estimators(type_filter='classifier')

print('sk version : ', sk.__version__)
print('모델의 개수 :', len(all))


#3,4. train, evaluate
for name, model in all:
    try:
        model = model()

        scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=kfold,
            scoring='accuracy'
        )

        print(
            name,
            'KFold acc :',
            np.round(scores, 4),
            '평균 acc :',
            round(np.mean(scores), 4)
        )

    except:
        print(name, '모델 오류')