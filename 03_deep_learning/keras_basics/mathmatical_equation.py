import numpy as np
#softmax
from math import exp

def softmax(input_vector):
    exponents = [exp(i) for i in input_vector]

    sum_of_exponents = sum(exponents)

    probabilities = [round(i / sum_of_exponents, 3) for i in exponents]

    return probabilities
print(softmax([-3, -2, -1, 0, 1, 2 ,3]))


#standardscaler

class MyStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x):
        # 각 feature별 평균
        self.mean_ = np.mean(x, axis=0)

        # 각 feature별 표준편차
        self.scale_ = np.std(x, axis=0)

        # 표준편차가 0이면 나누기 오류 방지
        self.scale_[self.scale_ == 0] = 1

        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

