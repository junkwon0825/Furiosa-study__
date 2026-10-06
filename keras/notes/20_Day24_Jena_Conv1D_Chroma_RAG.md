# Day 24 - Jena Conv1D / Chroma / RAG 기초

## 1. 오늘 배운 핵심

오늘은 크게 두 가지를 배웠다.

```text
1. Jena 시계열 데이터를 Conv1D로 모델링
2. Chroma를 이용한 RAG용 Vector DB / Vector Store 기초
```

---

# 2. Jena 시계열 데이터 → CNN(Conv1D)

기존에는 Jena 기후 데이터를 LSTM으로 처리했지만, 시계열 데이터는 `Conv1D`로도 모델링할 수 있다.

## 입력 shape

Jena 데이터의 입력 shape:

```text
(samples, timesteps, features)

(N, 144, 13)
```

- `N` : 샘플 개수
- `144` : timestep
- `13` : 각 timestep마다 들어가는 기상 feature

LSTM과 Conv1D 모두 기본적으로 **3차원 입력**을 사용한다.

```text
LSTM   : (N, T, F)
Conv1D : (N, T, F)
```

따라서 Conv1D를 사용할 때는 `x`를 4차원으로 reshape할 필요가 없다.

```python
# 필요 없음
x = x.reshape(-1, 144, 13, 1)
```

그대로:

```python
x.shape
# (N, 144, 13)
```

사용하면 된다.

---

## Conv1D 모델 예시

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout

model = Sequential()

model.add(
    Conv1D(
        64,
        kernel_size=3,
        input_shape=(144, 13),
        activation='relu',
        padding='same'
    )
)

model.add(
    Conv1D(
        32,
        kernel_size=3,
        activation='relu',
        padding='same'
    )
)

model.add(Dropout(0.2))

model.add(
    Conv1D(
        16,
        kernel_size=3,
        activation='relu',
        padding='same'
    )
)

model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
```

shape 흐름:

```text
(N, 144, 13)
↓
Conv1D(64)
(N, 144, 64)
↓
Conv1D(32)
(N, 144, 32)
↓
Conv1D(16)
(N, 144, 16)
↓
Dense(1)
(N, 144, 1)
```

`padding='same'`을 사용하면 timestep 길이 `144`가 유지된다.

---

## y reshape

정답 데이터가 원래:

```text
(N, 144)
```

이라면 모델 출력과 맞추기 위해:

```python
y = y.reshape(y.shape[0], y.shape[1], 1)
```

결과:

```text
(N, 144, 1)
```

로 맞춘다.

---

# 3. Chroma 설치

CMD 또는 터미널:

```bash
pip install langchain_chroma
pip install langchain-community
```

---

# 4. Chroma란?

**Chroma는 RAG에서 사용하기 편한 Vector DB / Vector Store이다.**

한 줄로 정리하면:

> 문서의 Embedding Vector와 원본 문서를 저장하고, 질문과 의미가 비슷한 문서를 검색해주는 저장소

전체 흐름:

```text
Document
↓
Chunking
↓
Embedding
↓
Chroma
↓
Similarity Search
↓
Retriever
↓
관련 문서
```

---

# 5. Character / Word / Token / Chunk

| 개념 | 뜻 | 예 |
|---|---|---|
| Character | 글자 하나 | `삼`, `성`, `A` |
| Word | 사람이 보는 단어 | `삼성전자` |
| Token | 모델이 처리하는 단위 | `삼성`, `전자` 등 |
| Chunk | 여러 문장/토큰을 묶은 문서 조각 | 약 300자짜리 문서 조각 |

```text
문자
= 글자 단위

Token
= AI 모델이 실제로 처리하는 단위

Chunk
= RAG 검색을 위해 문서를 잘라놓은 조각
```

---

# 6. Chunking

긴 문서를 그대로 Embedding하지 않고 작은 조각으로 나누는 과정이다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)
```

## chunk_size

```python
chunk_size=300
```

`RecursiveCharacterTextSplitter` 기본 설정에서는 대략 **300 문자 단위**로 문서를 나눈다.

## chunk_overlap

```python
chunk_overlap=100
```

앞 chunk와 뒤 chunk가 일정 부분 겹치도록 한다.

```text
Chunk 1
0 ---------------- 299

Chunk 2
        200 ---------------- 499
```

문맥이 chunk 경계에서 끊기는 것을 줄이기 위한 방법이다.

---

# 7. LangChain Document

LangChain에서 문서는 보통 `Document` 객체로 관리한다.

```text
Document
├── page_content
└── metadata
```

## page_content

실제 문서 내용

```python
doc.page_content
```

## metadata

문서의 추가 정보

```python
doc.metadata
```

예:

```python
{
    "source": "./_data/rag_data/samsung_outlook.txt"
}
```

---

# 8. 여러 txt 파일 불러오기

```python
from glob import glob
import os

path = './_data/rag_data/'

txt_files = glob(
    os.path.join(path, '*.txt')
)
```

`glob()`을 사용하면 폴더 안의 `.txt` 파일을 한 번에 가져올 수 있다.

---

# 9. TextLoader

```python
from langchain_community.document_loaders import TextLoader
```

txt 파일을 LangChain `Document` 객체로 변환한다.

```python
data = []

for text_file in txt_files:
    loader = TextLoader(
        text_file,
        encoding='utf-8'
    )
    data += loader.load()
```

전체 흐름:

```text
.txt
↓
TextLoader
↓
Document
```

---

# 10. 문서 문자 수 확인

```python
char_count = [
    len(doc.page_content)
    for doc in data
]

print(char_count)
```

예:

```text
[8158, 2049, 1898]
```

주의:

```text
문자 수 ≠ Token 수
```

---

# 11. 문서 Chunking

```python
texts = text_splitter.split_documents(data)
```

원본 Document를 여러 개의 작은 Document로 나눈다.

```text
원본 Document
↓
split_documents()
↓
Chunk Document 여러 개
```

각 chunk도 여전히 `Document` 객체이다.

---

# 12. Embedding

Embedding은 텍스트를 숫자 Vector로 변환하는 과정이다.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
)
```

예:

```text
"삼성전자의 창업자는 누구인가요?"
↓
Embedding
↓
[0.12, -0.31, 0.88, ...]
```

직접 확인:

```python
sample_text = "삼성전자의 창업자는 누구인가요?"
vector = embeddings.embed_query(sample_text)
print(len(vector))
```

---

# 13. Chroma 저장

```python
from langchain_chroma import Chroma
```

저장 위치:

```python
DB_PATH = './_db/Chroma12/'
```

저장:

```python
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma12',
)
```

내부 흐름:

```text
Chunk Document
↓
Embedding
↓
Vector
↓
Chroma 저장
```

Chroma에는 개념적으로 다음 정보들이 저장된다.

```text
ID
+
Document
+
Embedding Vector
+
Metadata
```

---

# 14. Similarity Search

```python
query = "삼성전자의 창업자는 누구인가요"

result = vector_store.similarity_search(query)
```

내부 흐름:

```text
질문
↓
Embedding
↓
Query Vector
↓
Chroma에 저장된 Vector와 비교
↓
가장 비슷한 문서 반환
```

중요:

> Chroma는 최종 답변을 만드는 것이 아니라 관련 문서를 찾아준다.

---

# 15. Retriever

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
```

Chroma Vector Store를 **검색기 형태**로 사용하는 것이다.

```text
Chroma
↓
as_retriever()
↓
Retriever
```

검색:

```python
aaa = retriever.invoke(query)
```

`k=2`이므로 가장 관련 있는 문서 2개를 가져온다.

---

# 16. Similarity Search와 Retriever 차이

직접 검색:

```python
vector_store.similarity_search(
    query,
    k=2
)
```

Retriever:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k":2}
)

retriever.invoke(query)
```

Retriever를 사용하는 이유는 이후 RAG Chain에 연결하기 편하기 때문이다.

