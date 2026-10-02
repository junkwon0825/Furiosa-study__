# Day23~24 정리 - RNN, Embedding, Vector DB, CNN→RNN, Reshape

## 1. 오늘 핵심 흐름

```text
텍스트
→ Tokenizer
→ Token ID
→ Padding
→ Embedding
→ RNN / LSTM
→ Dense
→ 분류 결과
```

이번 수업의 핵심은 **Embedding 뒤에 RNN/LSTM을 붙여서 순서가 있는 데이터를 처리하는 것**이다.

---

## 2. Embedding + RNN

### Embedding

```python
model.add(
    Embedding(
        input_dim=5000,
        output_dim=300,
        input_length=500
    )
)
```

의미:

```text
input_dim=5000
→ 사용할 단어 사전의 크기

output_dim=300
→ 단어 하나를 300차원 벡터로 표현

input_length=500
→ 한 문장을 500개 token 길이로 맞춤
```

입력과 출력 shape:

```text
입력
(N, 500)

↓ Embedding

(N, 500, 300)
```

즉 Embedding이 각 token ID를 의미를 학습할 수 있는 벡터로 변환한다.

### Embedding 뒤에 LSTM

```python
model.add(Embedding(input_dim=5000, output_dim=300, input_length=500))
model.add(LSTM(64))
```

shape 흐름:

```text
(N, 500)
↓
Embedding
↓
(N, 500, 300)
↓
LSTM(64)
↓
(N, 64)
```

LSTM 기준:

```text
(samples, timesteps, features)

(N, 500, 300)
```

- `500` = timestep
- `300` = timestep마다 들어가는 feature
- `64` = LSTM unit 수

---

## 3. Reuters 뉴스 분류

Reuters 데이터는 뉴스 기사를 **46개 카테고리**로 분류하는 다중분류 문제다.

```python
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=5000,
    test_split=0.2
)
```

shape:

```text
x_train : (8982,)
y_train : (8982,)

x_test  : (2246,)
y_test  : (2246,)
```

`x_train[i]`는 길이가 서로 다른 Python list다.

```text
뉴스기사 최대 길이 : 2376
최소 길이          : 13
평균 길이          : 약 145
```

---

## 4. Padding

RNN에 넣으려면 모든 문장의 길이를 동일하게 맞춰야 한다.

```python
x_train = pad_sequences(
    x_train,
    maxlen=500,
    padding='pre',
    truncating='pre'
)
```

결과:

```text
(8982,)
↓
Padding
↓
(8982, 500)
```

### padding

```text
padding='pre'
```

앞쪽에 0을 채운다.

```text
[10, 20, 30]

maxlen=5

→ [0, 0, 10, 20, 30]
```

### truncating

```text
truncating='pre'
```

너무 긴 문장은 앞부분을 자른다.

---

## 5. Reuters 모델

```python
model = Sequential()

model.add(Embedding(
    input_dim=5000,
    output_dim=300,
    input_length=500
))

model.add(LSTM(64))

model.add(Dense(96, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(46, activation='softmax'))
```

전체 흐름:

```text
Token ID
(N, 500)

↓ Embedding

(N, 500, 300)

↓ LSTM(64)

(N, 64)

↓ Dense

46개 클래스 확률
```

---

## 6. categorical vs sparse categorical

### One-Hot Encoding을 한 경우

```python
y_train = to_categorical(y_train, 46)
```

shape:

```text
(N,)
↓
(N, 46)
```

loss:

```python
loss='categorical_crossentropy'
```

### One-Hot Encoding을 안 한 경우

y가 그대로 정수 label이면:

```text
0
1
2
...
45
```

```python
loss='sparse_categorical_crossentropy'
```

을 사용할 수 있다.

```text
y가 정수 label
→ sparse_categorical_crossentropy

y가 One-Hot
→ categorical_crossentropy
```

**핵심:** `sparse_categorical_crossentropy`를 사용하면 y를 One-Hot Encoding하지 않아도 된다.

---

## 7. DNN / CNN / RNN / Embedding 비교

| 구분 | DNN | CNN | RNN | Embedding |
|---|---|---|---|---|
| 정체 | 신경망 구조 | 신경망 구조 | 신경망 구조 | 벡터 표현 레이어 |
| 잘하는 것 | 일반 수치 데이터 | 이미지 / 공간 패턴 | 순서 / 시계열 / 문장 | ID → Dense Vector |
| 핵심 | Dense 연결 | Filter / Kernel | 이전 상태 기억 | 의미 벡터 표현 |
| 대표 입력 | `(N,F)` | `(N,H,W,C)` | `(N,T,F)` | `(N,T)` |
| 대표 출력 | `(N,units)` | Feature Map | Hidden State | `(N,T,E)` |
| 예시 | 집값, 일반 분류 | 이미지, 차선 | 문장, 센서 시계열 | 단어, Token |

> 주의: `Flatten`은 특정 모델에 반드시 필요한 고정 규칙은 아니다. CNN에서 Dense로 연결할 때 자주 사용하고, Embedding 출력은 목적에 따라 RNN/LSTM/Flatten 등으로 연결할 수 있다.

---

## 8. CNN → RNN 변환

이번 과제의 핵심:

```text
CNN 입력 = 4차원
RNN/LSTM 입력 = 3차원
```

CNN:

```python
x_train = x_train.reshape(-1, 2, 4, 1)
```

shape:

```text
(samples, height, width, channels)
(N, 2, 4, 1)
```

RNN/LSTM:

```python
x_train = x_train.reshape(-1, 8, 1)
```

shape:

```text
(samples, timesteps, features)
(N, 8, 1)
```

### 예제

원본:

```text
(N, 8)
```

CNN:

```python
x_train = x_train.reshape(-1, 2, 4, 1)
```

```text
(N, 2, 4, 1)
```

LSTM:

```python
x_train = x_train.reshape(-1, 8, 1)
```

```text
(N, 8, 1)
```

---

## 9. 회귀 모델 마지막 출력

회귀 문제에서는 마지막 출력:

```python
model.add(Dense(1))
```

을 사용한다.

예:

```text
집값
자전거 대여량
당뇨 수치
```

잘못된 예:

```python
Dense(1, activation='softmax')
```

출력이 1개인 softmax는 항상 1이 되기 때문에 회귀에 사용할 수 없다.

```text
회귀
→ Dense(1)

이진분류
→ Dense(1, sigmoid)

다중분류
→ Dense(class개수, softmax)
```

---

## 10. SimpleRNN / LSTM 입력 Shape

RNN 계열 입력은 기본적으로 3차원이다.

```text
(samples, timesteps, features)
```

예:

```text
(1000, 20, 5)

1000 = 데이터 개수
20   = timestep
5    = timestep마다 들어가는 feature
```

### RNN을 여러 층 연결할 때

```python
model.add(
    SimpleRNN(
        128,
        return_sequences=True,
        input_shape=(20, 5)
    )
)

model.add(SimpleRNN(64))
```

첫 번째 RNN:

```text
(N, 20, 5)
↓
(N, 20, 128)
```

두 번째 RNN:

```text
(N, 20, 128)
↓
(N, 64)
```

RNN 뒤에 RNN을 또 붙이려면 앞 RNN에:

```python
return_sequences=True
```

를 사용한다.

---

## 11. Reshape Layer

전처리 단계에서 NumPy로:

```python
x_train = x_train.reshape(-1, 8, 1)
```

할 수도 있지만 모델 내부에서 shape을 바꿀 수도 있다.

```python
from tensorflow.keras.layers import Reshape
```

예:

```python
model = Sequential()

model.add(
    Reshape(
        (8, 1),
        input_shape=(8,)
    )
)

model.add(LSTM(64))
```

shape:

```text
입력
(N, 8)

↓ Reshape

(N, 8, 1)

↓ LSTM

(N, 64)
```

즉:

```python
x_train.reshape(...)
```

는 **모델 밖에서 shape 변경**,

```python
model.add(Reshape(...))
```

는 **모델 안에서 shape 변경**이다.

### Reshape 주의점

원소 개수는 같아야 한다.

가능:

```text
8
→ 8 × 1
```

가능:

```text
784
→ 28 × 28 × 1
```

불가능:

```text
784
→ 20 × 20
```

왜냐하면:

```text
784 ≠ 400
```

이기 때문이다.

> Reshape는 데이터를 새로 만드는 것이 아니라 **값은 그대로 두고 모양만 바꾼다.**

---

## 12. Vector DB / Vector Store

RAG에서 문서를 검색하려면 문서를 Embedding Vector로 바꿔 저장한다.

```text
Document
↓
Chunking
↓
Embedding
↓
Vector DB
```

질문이 들어오면:

```text
Question
↓
Embedding
↓
질문 Vector
↓
Vector Search
↓
관련 문서 검색
↓
LLM
↓
Answer
```

Vector DB의 역할:

> **문서의 Embedding Vector를 저장하고 질문 Vector와 비슷한 Vector를 빠르게 찾아준다.**

---

## 13. Chroma

Chroma는 Vector DB / Vector Store다.

주요 역할:

```text
Embedding Vector 저장
+
원본 Document 저장
+
Metadata 관리
+
유사도 검색
```

LangChain에서는 Chroma를 wrapping해서 편하게 사용할 수 있다.

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()
```

---

## 14. FAISS

FAISS:

```text
Facebook AI Similarity Search
```

Meta에서 만든 **벡터 검색 라이브러리 / 검색 엔진**이다.

핵심 역할:

> **대규모 Vector에서 비슷한 Vector를 매우 빠르게 찾는 것**

즉 FAISS는 Vector 저장소 전체 기능보다 **Vector Search 자체에 더 집중**한다.

---

## 15. Chroma vs FAISS

| 구분 | Chroma | FAISS |
|---|---|---|
| 성격 | Vector DB / Vector Store | Vector Search Library |
| Vector 저장 | O | O |
| 유사도 검색 | O | O |
| 문서 관리 | O | 기본 기능은 제한적 |
| Metadata | O | 직접 관리 필요 |
| LangChain 연결 | 쉬움 | 가능 |
| 강점 | RAG 문서 관리 | 빠른 벡터 검색 |
| 사용 느낌 | DB에 가까움 | 검색 엔진에 가까움 |

쉽게 기억:

```text
FAISS
= 벡터 찾기 전문

Chroma
= 벡터 + 문서 + Metadata 관리
```

---

## 16. Chroma와 FAISS를 언제 사용할까?

### Chroma

```text
PDF
사내 문서
자동차 매뉴얼
기술 문서
FAQ
```

처럼 **문서 자체를 함께 관리하면서 RAG를 만들고 싶을 때** 편하다.

### FAISS

```text
Vector 수가 많음
↓
빠른 유사도 검색이 중요
```

할 때 유용하다.

---

## 17. LangChain에서 Vector Store 위치

전체 RAG 구조:

```text
Document
   ↓
Chunking
   ↓
Embedding
   ↓
Chroma / FAISS
   ↓
Retriever
   ↓
Prompt
   ↓
LLM
   ↓
Parser
   ↓
Answer
```

즉:

```text
Embedding
→ 숫자 Vector 생성

Chroma / FAISS
→ Vector 저장 및 검색

Retriever
→ 관련 Document 가져오기

LLM
→ 검색된 Document를 보고 답변 생성
```

---

## 18. Embedding과 Vector DB의 차이

```text
Embedding
= 데이터를 Vector로 변환

Vector DB
= 만들어진 Vector를 저장하고 검색
```

예:

```text
"자동차 엔진오일 교체 주기"

↓ Embedding

[0.12, -0.41, 0.88, ...]

↓ Chroma

저장
```

질문:

```text
"엔진오일은 언제 바꿔?"
```

↓

```text
질문 Vector 생성
↓
Chroma / FAISS 검색
↓
비슷한 문서 반환
```

---

## 19. TensorFlow → PyTorch

TensorFlow/Keras에서 이미 배운 개념:

```text
Tensor
Layer
Dense
CNN
RNN
LSTM
Embedding
Loss
Optimizer
Epoch
Batch
Backpropagation
```

은 PyTorch에서도 거의 동일하다.

달라지는 것은 주로 **코드를 작성하는 방식**이다.

> TensorFlow/Keras에서 딥러닝 구조를 이해해두면 PyTorch를 배울 때 훨씬 쉽다.

---

## 20. 오늘 기억해야 할 핵심

```text
1. Embedding은 Token ID를 Dense Vector로 바꾼다.

2. Embedding 출력:
   (N,T) → (N,T,E)

3. RNN/LSTM 입력은 3차원:
   (samples, timesteps, features)

4. CNN 입력은 보통 4차원:
   (N,H,W,C)

5. CNN → RNN 변환 시
   4차원 → 3차원으로 reshape한다.

6. Reshape는 값은 그대로 두고 shape만 변경한다.

7. sparse_categorical_crossentropy를 쓰면
   y One-Hot Encoding을 생략할 수 있다.

8. FAISS는 Vector Search에 집중한다.

9. Chroma는 Vector + Document + Metadata를 관리한다.

10. Embedding → Vector DB → Retriever → LLM이
    RAG의 핵심 흐름이다.
```

