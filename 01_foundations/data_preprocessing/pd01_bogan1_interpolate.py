"""
결측치 처리
1. 삭제 - 행 또는 열 
2. 임의의 값
    - 0 : fillna
    - 평균 : mean (특히 이상치 조심)
    - 중위 : median
    - 앞값 : ffill
    - 뒷값 : bfill
    - 특정값 : 777(조건 보고 넣는게 낫다)
    - 최빈값
    - 기타 등등...
3. interpolate = 보간 (알려진 데이터 점 집합의 범위 내에 새 데이터 점을 추가하는 기법)
4. 모델 : .predict(값을 예측해서), (전 다른 모델 사용)
5. 부스팅(트리) 계열 모델 : 통상 결측치, 이상치에 대해 영향을 덜 받는다. (자유롭다)
6. 스케일링(로브스트등....) -이상치에 자유로운놈..
"""


import pandas as pd
import numpy as np

dates = [
    '11/11/2026','12/11/2026','13/11/2026',
    '14/11/2026','15/11/2026','16/11/2026',
    '17/11/2026','18/11/2026','19/11/2026',
]
# dates = pd.to_datetime(dates)
dates = pd.to_datetime(dates, format="%d/%m/%Y")
print(dates)


print("===============================")
ts = pd.Series([0,2, np.nan, np.nan, 8 ,10, np.nan, 14,16], index=dates)
print(ts)

print("===============================")

ts = ts.interpolate()
print(ts)

# 중간값들은 linear로 채워짐.
# 마지막에 결측치일 경우 ffill (이전값으로 채워짐)
# 첫번째 결측치는 처리 안됨.







