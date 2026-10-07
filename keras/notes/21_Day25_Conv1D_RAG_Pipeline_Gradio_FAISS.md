# Day 25 - Conv1D / RAG Pipeline / Gradio Chatbot / FAISS

## 1. 오늘 배운 핵심

```text
1. Dense / Conv1D / Conv2D / RNN / Embedding 입력·출력 shape 비교
2. RAG Pipeline 구조
3. Chroma + LangChain + Gradio를 이용한 RAG 챗봇
4. FAISS 기본 개념
```
---

# 3. Conv1D

Conv1D는 시계열 데이터에서 자주 사용할 수 있다.

입력 shape:

```text
(N, T, F)
```

예:

```text
(N, 144, 13)
```

- `144` = timestep
- `13` = feature

예시:

```python
model.add(
    Conv1D(
        64,
        kernel_size=3,
        input_shape=(144, 13),
        padding='same',
        activation='relu'
    )
)
```

출력:

```text
(N, 144, 64)
```

`padding='same'`이면 timestep 길이를 유지할 수 있다.

## Conv1D와 RNN 공통점

```text
Conv1D : (N, T, F)
RNN    : (N, T, F)
LSTM   : (N, T, F)
GRU    : (N, T, F)
```

차이:

```text
Conv1D
= 주변 구간의 패턴 추출

RNN/LSTM/GRU
= 시간 순서와 이전 정보 기억
```

---

# 4. RAG Pipeline

Pipeline은 여러 단계를 순서대로 연결해서 하나의 처리 흐름으로 만드는 것이다.

```text
입력
↓
여러 처리 단계
↓
최종 출력
```

RAG에서는:

```text
query
↓
retriever
↓
relevant document
↓
prompt 생성
↓
model
↓
answer
```

즉 질문이 들어오면 관련 문서를 검색하고, 검색된 문서와 질문을 Prompt로 만든 뒤 LLM이 최종 답변을 생성한다.

---

# 5. Chroma 기반 RAG 구조

기존에 저장한 Chroma Vector Store를 불러온다.

```python
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma12',
)
```

Retriever 생성:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
```

의미:

```text
질문
↓
Retriever
↓
Chroma
↓
가장 관련 있는 문서 2개
```

---

# 6. LLM 연결

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5-nano-chat",
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)
```

`temperature=0`:

```text
답변을 비교적 일관되고 결정적으로 생성
```

---

# 7. Prompt 만들기

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요.
컨텍스트에 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다."라고 말씀해 주세요.

컨텍스트 : {context}
질문 : {input}
답변:
""")
```

여기서:

```text
{context}
= Retriever가 검색한 관련 문서

{input}
= 사용자의 질문
```

---

# 8. Document Chain

```python
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

docu_chain = create_stuff_documents_chain(
    model,
    prompt
)
```

개념적으로:

```text
검색된 문서
+
Prompt
↓
LLM
↓
Answer
```

---

# 9. Retrieval Chain

```python
from langchain_classic.chains import create_retrieval_chain

rag_chain = create_retrieval_chain(
    retriever,
    docu_chain
)
```

전체 구조:

```text
Question
↓
Retriever
↓
Relevant Documents
↓
Prompt
↓
LLM
↓
Answer
```

---

# 10. RAG Chain 실행

```python
query = "삼성전자의 창업자는 누구인가요?"

response = rag_chain.invoke(
    {"input": query}
)
```

반환값은 Dictionary 형태이다.

```python
print(response.keys())
```

예:

```text
dict_keys(['input', 'context', 'answer'])
```

주요 값:

```python
response['input']
response['context']
response['answer']
```

---

# 11. Gradio 설치

가상환경에서:

```bash
pip install gradio
```

---

# 12. Gradio 챗봇

```python
import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke(
        {"input": message}
    )
    return response['answer']

demo = gr.ChatInterface(
    fn=answer_invoke,
    title='감자'
)

demo.launch()
```

전체 흐름:

```text
사용자
↓
Gradio ChatInterface
↓
answer_invoke()
↓
rag_chain.invoke()
↓
Retriever
↓
관련 문서
↓
Prompt
↓
LLM
↓
response['answer']
↓
Gradio 화면 출력
```

## message / history

```python
def answer_invoke(message, history):
```

- `message` = 현재 사용자가 입력한 질문
- `history` = 이전 대화 기록

현재 코드에서는 `history`를 직접 사용하지 않는다.

---

# 13. FAISS

FAISS:

```text
Facebook AI Similarity Search
```

벡터들 중에서 가장 비슷한 벡터를 빠르게 찾아주는 검색 라이브러리이다.

설치:

```bash
pip install faiss-cpu
```

---

# 14. FAISS가 하는 일

```text
질문
"삼성전자 사업 전망 알려줘"
↓
Embedding
↓
Query Vector
↓
FAISS
↓
가장 가까운 Document Vector 검색
↓
관련 문서 반환
```

즉 많은 벡터 중에서 질문 벡터와 가장 가까운 벡터를 빠르게 찾는다.

---

# 15. Chroma와 FAISS 비교

| 구분 | Chroma | FAISS |
|---|---|---|
| 핵심 역할 | Vector DB / Vector Store | Vector Similarity Search |
| 벡터 검색 | 가능 | 매우 강함 |
| 문서 관리 | 편리 | 별도 관리가 더 필요 |
| Metadata | 지원 | 직접 관리 필요 |
| RAG 연동 | 편리 | 가능 |
| 대규모 유사도 검색 | 가능 | 특히 강함 |

쉽게 정리하면:

```text
FAISS
= 벡터 검색 전문 엔진

Chroma
= 벡터 + 문서 + metadata까지 관리하는 Vector DB
```

---

# 16. 대규모 유사도 검색

벡터가 적으면 모든 벡터와 직접 비교할 수 있다.

```text
질문 Vector
↓
문서 Vector 1 비교
문서 Vector 2 비교
문서 Vector 3 비교
...
```

하지만 벡터가 수백만 개 이상이면 모든 벡터를 하나씩 비교하는 것은 부담이 커진다.

FAISS는 벡터를 Index로 구성해서 빠르게 가까운 벡터를 검색할 수 있다.

```text
수많은 Vector
↓
Index
↓
Query Vector
↓
Nearest Vector Search
↓
TOP K 반환
```

---

# 17. 오늘 전체 흐름

```text
[딥러닝]

Dense
Conv1D
Conv2D
RNN / LSTM / GRU
Embedding

↓

입력 shape 이해


[RAG]

Question
↓
Embedding
↓
Vector Store
↓
Retriever
↓
Relevant Documents
↓
Prompt
↓
LLM
↓
Answer


[서비스]

RAG Chain
↓
Gradio
↓
Chatbot


[Vector Search]

Chroma
or
FAISS
```

---

# 18. 오늘 핵심 용어

```text
Conv1D
= 3차원 시계열 데이터에서 1차원 방향 패턴 추출

Pipeline
= 여러 처리 단계를 순서대로 연결한 전체 흐름

Retriever
= Vector Store에서 관련 문서를 검색하는 검색기

RAG
= 검색한 문서를 근거로 LLM이 답변하도록 하는 구조

Gradio
= Python 모델이나 AI 기능을 웹 UI로 빠르게 보여주는 도구

FAISS
= 빠른 Vector Similarity Search 라이브러리

Chroma
= 문서 + Vector + Metadata를 함께 관리하기 편한 Vector Store
```
