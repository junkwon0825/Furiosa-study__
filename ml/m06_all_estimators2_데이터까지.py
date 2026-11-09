import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators
import time
import warnings
warnings.filterwarnings('ignore')

#1. data
california = fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)

datasets = [california, diabetes]
data_name = ['california', 'diabetes']


#2. model
all = all_estimators(type_filter='regressor')

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

start = time.time()

for index, value in enumerate(datasets):
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, train_size=0.8, random_state=123
    )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("=================", data_name[index], "============")

    model_name = []
    model_r2 = []

    for name, model in all:
        try:
            model = model()

            #3. train
            scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
            y_pred = cross_val_predict(model, x_train, y_train, cv=kfold)

            #4. evaluate
            r2 = np.round(r2_score(y_train, y_pred), 4)

            model_name.append(name)
            model_r2.append(r2)

        except:
            continue

    model_r2 = np.array(model_r2)
    max_index = np.argmax(model_r2)

    print('최고의 모델 :', model_name[max_index], 'R2:', model_r2[max_index])

end = time.time()

print("걸린시간 :", round(end - start, 2), '초')