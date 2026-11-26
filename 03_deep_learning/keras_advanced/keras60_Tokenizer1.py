from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
text = ' 나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 마구 마구 먹음'

token = Tokenizer() #클래스 정의한 애를 부르는 말 instance(객체)
token.fit_on_texts([text])

print(token.word_index)
#{'진짜': 1, '매우': 2, '마구': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '먹음': 8}
# 많이 사용하는 순서 빈도순으로 번호 높음 빈도수같으면 앞에 애가 우선순위
print(token.word_counts)
#OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('마구', 2), ('먹음', 1)])
#여기까지의 문제는 나는은 진짜의 2배가아님 수치화하면 -> 원핫인코딩
x = token.texts_to_sequences([text])  # text 수치화  
print(x)
#[[4, 5, 1, 1, 2, 2, 6, 7, 3, 3, 8]]
x=x[0]
print(len(x))

#원핫 인코딩 3가지 만들기#
# #1. pandas
# x = pd.get_dummies(x, dtype=int)
# print(x)
# print(x.shape) #(11,8)

#2. sklearn
from sklearn.preprocessing import OneHotEncoder #onehotencoder가져왔으니까 정의
ohe = OneHotEncoder(sparse_output=False) # sparse->혼돈행렬 형태로 나옴 그래서 sparse_output=False
x = np.array(x).reshape(-1,1)
x = ohe.fit_transform(x)
print(x)
print(x.shape) #(11,8)

# # #3. keras
# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# print(x.shape) #(11, 9)
