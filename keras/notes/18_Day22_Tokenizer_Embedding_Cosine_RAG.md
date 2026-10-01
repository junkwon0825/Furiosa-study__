# Day22 - Tokenizer, Padding, One-Hot Encoding, Embedding, Cosine Similarity

## 1. 오늘의 전체 흐름

오늘은 텍스트 데이터를 모델이 처리할 수 있는 숫자 형태로 바꾸는 과정을 정리했다.

```text
Text
↓
Tokenizer
↓
Token IDs
↓
Padding
↓
One-Hot Encoding 또는 Embedding
↓
RNN / LSTM / Transformer
```

RAG 쪽에서는 다음 흐름으로 연결된다.

```text
문장 / 문서 Chunk
↓
Embedding Model
↓
Vector
↓
Vector DB
↓
유사도 검색
↓
Retriever
↓
LLM
```

---

## 2. Tokenizer

Tokenizer는 문장을 잘게 나누고 각 토큰에 번호를 붙이는 과정이다.

```text
"LKA는 차량이 차선을 유지하도록 조향을 보조한다."

↓ Tokenizer

[LKA, 는, 차량, 이, 차선, 을, ...]

↓ Token ID

[102, 51, 783, 22, ...]
```

GPT 같은 LLM도 글자를 그대로 계산하는 것이 아니라 토큰 단위로 처리한다.

```text
Text
↓
Tokenizer
↓
Token IDs
↓
Transformer
↓
다음 Token 확률
```

### 핵심

> Token ID는 단순한 번호표이며 숫자의 크기 자체에는 의미가 없다.

예를 들어 단어 ID가 `3`, `20`이라고 해서 20번 단어가 3번 단어보다 크거나 중요한 것이 아니다.

---

## 3. Python List와 NumPy Array

| 표기 | 자료형 |
|---|---|
| `[]` | list |
| `[[]]` | 중첩 list |
| `{}` | dict |
| `()` | tuple |

Python list는 값을 담는 기본 자료형이고, NumPy array는 수치 계산과 행렬 연산을 하기 좋은 자료형이다.

```python
a = [1, 2, 3]

import numpy as np
b = np.array([1, 2, 3])

print(b.shape)
# (3,)
```

벡터 여러 개를 쌓으면 행렬이 된다.

```python
x = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(x.shape)
# (2, 3)
```

---

## 4. 여러 문장 합치기

Tokenizer 결과가 여러 개의 문장으로 나뉘어 있을 때 필요에 따라 `np.concatenate()`로 하나의 배열처럼 이어 붙일 수 있다.

```python
x = token.texts_to_sequences([text1, text2])
x = np.concatenate(x)
```

다만 문장 구조를 유지해야 하는 RNN/LSTM 입력에서는 무조건 합치기보다 Padding 이후 shape를 유지하는 것이 중요하다.

---

## 5. Padding

문장은 길이가 제각각이다.

```text
[2, 3]
[1, 4]
[1, 5, 6]
[10, 11, 12, 13, 14]
```

이 상태로는 동일한 shape의 행렬로 만들기 어렵다.

그래서 `pad_sequences()`를 사용하여 문장 길이를 맞춘다.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_x = pad_sequences(
    x,
    maxlen=5,
    padding='post',
    truncating='post'
)
```

옵션 의미:

```text
padding='post'
→ 뒤쪽에 0을 붙인다.

truncating='post'
→ 너무 긴 문장은 뒤쪽을 자른다.
```

예:

```text
[2, 3]
↓
[2, 3, 0, 0, 0]
```

### 핵심

> Padding은 서로 다른 길이의 문장을 같은 길이로 맞추기 위해 빈 자리를 0으로 채우는 과정이다.

Tokenizer에서는 `0`을 실제 단어 ID 대신 padding 같은 특별한 용도로 남겨두는 경우가 많다.

---

## 6. One-Hot Encoding

Tokenizer가 만든 숫자는 단순 ID다.

예:

```text
재미있다 → 3
최고예요 → 4
바보 → 25
```

이 숫자의 크기 자체에는 의미가 없기 때문에 범주형 데이터임을 명확하게 표현하기 위해 One-Hot Encoding을 사용할 수 있다.

```text
0 → [1, 0, 0, 0, ...]
1 → [0, 1, 0, 0, ...]
2 → [0, 0, 1, 0, ...]
```

### sklearn OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

x = padded_x.reshape(-1, 1)
x = ohe.fit_transform(x)
x = x.reshape(15, 5, 32)
```

shape 흐름:

```text
(15, 5)
↓ reshape
(75, 1)
↓ One-Hot Encoding
(75, 32)
↓ 다시 문장 구조로 reshape
(15, 5, 32)
```

LSTM 입력 형식:

```text
(samples, timesteps, features)

(15, 5, 32)
```

- 15: 문장 개수
- 5: 문장당 최대 토큰 개수
- 32: 각 토큰의 One-Hot feature 수

### 왜 바로 `(15, 5, 32)`로 reshape할 수 없는가?

`reshape()`는 데이터 개수를 늘리지 않고 모양만 바꾼다.

```text
(15, 5)
= 75개 숫자

(15, 5, 32)
= 2400개 숫자
```

따라서 중간에 One-Hot Encoding을 통해 각 토큰을 여러 개의 값으로 확장해야 한다.

---

## 7. One-Hot Encoding의 문제점

One-Hot Encoding은 단어 종류가 많아질수록 벡터 길이가 매우 커진다.

예를 들어 vocabulary가 100,000개라면:

```text
단어 하나
→ 100,000차원 벡터
```

대부분이 0이므로 매우 비효율적이다.

또한 단어 간 의미 관계를 표현하지 못한다.

```text
재미있다
→ [0, 0, 1, 0, ...]

재밌네요
→ [0, 0, 0, 0, 1, ...]
```

두 단어가 의미상 비슷하더라도 One-Hot에서는 단순히 서로 다른 category일 뿐이다.

---

## 8. Embedding

Embedding은 토큰 ID를 더 작고 밀집된 실수 벡터로 바꾼다.

예:

```text
3
↓
Embedding
↓
[0.21, -0.52, 0.63, 0.11, ...]
```

예를 들어:

```python
Embedding(
    input_dim=32,
    output_dim=8,
    input_length=5
)
```

이면 단어 하나가 8차원 벡터로 표현된다.

```text
output_dim = 8
→ feature 수 = 8
→ 8차원 벡터
```

shape:

```text
입력
(15, 5)

↓ Embedding(output_dim=8)

(15, 5, 8)
```

LSTM 입력 관점:

```text
(samples, timesteps, features)
= (15, 5, 8)
```

### Embedding Parameter 계산

```python
Embedding(input_dim=31, output_dim=100)
```

이면:

```text
31 × 100 = 3100
```

개의 학습 가능한 parameter가 있다.

Embedding은 내부적으로 토큰별 벡터를 저장하는 행렬을 가지고 있다고 볼 수 있다.

```text
token 0  → 100차원 벡터
token 1  → 100차원 벡터
token 2  → 100차원 벡터
...
```

### One-Hot과 Embedding 비교

```text
One-Hot
→ 크고 대부분 0
→ 의미 관계 없음

Embedding
→ 더 작은 dense vector
→ 학습을 통해 의미 있는 표현 가능
```

---

## 9. Embedding + SimpleRNN

예:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(input_dim=31, output_dim=100, input_length=5))
model.add(SimpleRNN(10))

model.summary()
```

Embedding 출력:

```text
(None, 5, 100)
```

의미:

- None: batch size
- 5: timestep
- 100: 각 토큰의 embedding feature

Embedding parameter:

```text
31 × 100 = 3100
```

SimpleRNN(10)의 parameter:

```text
입력 weight
100 × 10 = 1000

순환 weight
10 × 10 = 100

bias
10

총합
1000 + 100 + 10 = 1110
```

공식:

```text
SimpleRNN Param
= units × (input_features + units + 1)
```

현재 예:

```text
10 × (100 + 10 + 1)
= 1110
```

---

## 10. Embedding Model

RAG에서 말하는 Embedding Model은 문장이나 문서의 의미를 벡터로 바꿔주는 모델이다.

```text
"ACC는 앞차와의 거리를 유지한다."

↓ Embedding Model

[0.13, -0.21, 0.77, ...]
```

Keras의 `Embedding()` 레이어는 모델 내부에서 토큰별 벡터를 학습하는 층이고, RAG의 Embedding Model은 보통 이미 학습된 모델을 이용해 문장이나 문서 chunk 전체를 의미 벡터로 변환한다.

---

## 11. Vector Store / Vector DB

Embedding Model로 만든 벡터를 저장하는 곳이다.

```text
문서
↓
Embedding Model
↓
Vector
↓
Vector DB
```

RAG에서는 사용자의 질문도 같은 Embedding Model로 벡터화한 뒤 저장된 문서 벡터들과 비교한다.

---

## 12. Cosine Similarity

코사인 유사도는 두 벡터의 방향이 얼마나 비슷한지를 측정한다.

```text
비슷한 방향
→ cosine similarity가 1에 가까움

관계가 적은 방향
→ 0에 가까움

반대 방향
→ -1에 가까움
```

예:

```text
A = "자동차가 앞차와 거리를 유지한다."
B = "ACC는 선행 차량과 안전거리를 유지한다."
C = "오늘 삼겹살을 먹었다."
```

각 문장을 embedding하면:

```text
A → vector A
B → vector B
C → vector C
```

A와 B는 의미가 비슷하므로 cosine similarity가 높게 나올 가능성이 있고, A와 C는 낮게 나올 가능성이 있다.

### 핵심

> Embedding Model은 문장의 의미를 벡터 좌표로 바꾸고, Cosine Similarity는 벡터 방향을 비교하여 의미상 비슷한 문장을 찾는다.

---

## 13. RAG와 연결

RAG 검색 과정:

```text
[문서]
↓
Chunking
↓
Embedding Model
↓
Vector DB 저장


[사용자 질문]
↓
Embedding Model
↓
Question Vector
↓
Vector DB에서 유사도 검색
↓
관련 문서 검색
↓
Retriever
↓
LLM
↓
답변
```

즉 오늘 배운 Embedding과 Cosine Similarity는 RAG의 검색 단계와 직접 연결된다.

---

## 14. 오늘 마지막 LangChain Embedding 코드

```python
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = "삼성전자의 창업주는 누구인가요?"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,
)

vector = embeddings.embed_query(prompt)

print(vector)
print("=================================")
print("임베딩 벡터의 차원:", len(vector))
```

이 코드의 흐름:

```text
문장
"삼성전자의 창업주는 누구인가요?"

↓ embed_query()

Embedding Model

↓
5차원 Vector

[x1, x2, x3, x4, x5]
```

`dimensions=5`를 사용하여 실습에서는 벡터의 차원을 5로 조절했다.

---

# 오늘의 핵심 정리

```text
Tokenizer
= 문장을 토큰으로 나누고 ID를 붙인다.

Padding
= 문장 길이를 동일하게 맞춘다.

One-Hot Encoding
= Token ID를 범주형 벡터로 표현한다.

Embedding
= Token이나 문장을 작고 밀집된 의미 벡터로 표현한다.

Vector DB
= Embedding Vector를 저장한다.

Cosine Similarity
= Vector끼리 방향을 비교해 의미상 가까운 데이터를 찾는다.
```

최종적으로:

```text
Text
↓
Tokenizer
↓
Token IDs
↓
Padding
↓
Embedding
↓
Vector
↓
Vector DB
↓
Similarity Search
↓
Retriever
↓
LLM
```

