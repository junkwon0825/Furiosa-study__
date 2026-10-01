from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
text1 = ' 나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 마구 마구 먹음'
text2 = ' 개똥이는 선생을 괴롭힌다. 말똥이는 못생겼다. 길동이는 마구 마구 더 잘생겼다'

token = Tokenizer() #클래스 정의한 애를 부르는 말 instance(객체)
token.fit_on_texts([text1, text2])

print(token.word_index)

x = token.texts_to_sequences([text1,text2])  # text 수치화  
print(x)
#[[4, 5, 2, 2, 3, 3, 6, 7, 1, 1, 8], [9, 10, 11, 12, 13, 14, 1, 1, 15, 16]]

#numpy concatenate 엮어서 하나로 원핫인코딩.
x = np.concatenate(x)
print(x) #[ 4  5  2  2  3  3  6  7  1  1  8  9 10 11 12 13 14  1  1 15 16]
print(x.shape) #(21,)

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
print(x.shape) #(21, 16)

# # #3. keras
# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# print(x.shape) #(11, 9)


