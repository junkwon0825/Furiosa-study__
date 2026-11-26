# 02. Machine Learning

## 학습 내용

scikit-learn과 XGBoost를 사용해 분류·회귀 모델 비교, 교차검증, feature importance, hyperparameter search를 실습했습니다.

## 핵심 개념

- `KFold`, `StratifiedKFold`, `cross_val_score`, `cross_val_predict`
- `all_estimators`를 이용한 모델 후보 비교
- Decision Tree / Random Forest feature importance
- XGBoost classifier/regressor
- `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`
- 분류: Accuracy 중심 평가
- 회귀: R²와 회귀 score 중심 평가

## 대표 파일

| 파일 | 내용 |
|---|---|
| `code/m01_feature_importances*.py` | Tree 계열 및 XGBoost feature importance 비교 |
| `code/m02_kfold_*.py` | 데이터 유형별 KFold/StratifiedKFold |
| `code/m04_all_estimators*.py` | classifier/regressor 후보 탐색 |
| `code/m06_all_estimators2_데이터까지.py` | California/Diabetes 회귀 모델을 R²로 비교한 후속 예제 |
| `code/m08_GridSearchCV_*.py` | XGBoost Grid Search |
| `code/m09_RandomizedSearchCV*.py` | XGBoost Randomized Search |
| `code/m10_HalvingGridSearchCV_00.py` | successive halving search |

초기 `m04_all_estimators2_regressor.py`에는 분류 데이터와 회귀 estimator를 섞어 비교한 학습 단계가 남아 있습니다. 이를 삭제하거나 결과를 꾸미지 않고, 이후 `m06_all_estimators2_데이터까지.py`에서 회귀 데이터와 R² 평가로 정리된 흐름을 대표 예제로 명시했습니다.
