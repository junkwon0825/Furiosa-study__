# Day21 - RAG 시작, LangChain 개발환경, LCEL 기본

## 1. 오늘 학습 내용

오늘부터 RAG 학습을 시작했다.

- RAG 기본 개념
- 새로운 가상환경 구성
- LangChain / langchain_openai 설치
- VS Code 인터프리터 설정
- API Key / Base URL 환경변수 연결
- LangChain 기본 구조
- PMO / PLP 구조
- LCEL 기본 문법
- Prompt → Model 연결
- `chain.invoke()` 실행
- 앞으로의 RAG → Agent → MCP → NPU Serving 학습 흐름

---

## 2. RAG란?

RAG는 `Retrieval-Augmented Generation`의 약자이다.

```text
Retrieval
검색

+

Augmented
정보를 추가해서

+

Generation
LLM이 답변 생성
```

즉, 외부 문서나 데이터를 검색해서 관련 정보를 찾고, 그 정보를 LLM에게 함께 전달해 답변을 생성하게 하는 구조이다.

일반 LLM:

```text
사용자 질문
    ↓
   LLM
    ↓
   답변
```

RAG:

```text
사용자 질문
    ↓
관련 정보 검색
    ↓
검색된 정보 + 질문
    ↓
   LLM
    ↓
   답변
```

---

## 3. RAG 기본 구조

```text
문서
 ↓
Chunking
 ↓
Embedding
 ↓
Vector DB
 ↓
Retriever
 ↓
Prompt
 ↓
LLM
 ↓
Answer
```

앞으로 배우게 될 주요 구성 요소:

```text
Document
Chunk
Embedding
Vector DB
Retriever
Prompt
LLM
Parser
Tool
Agent
```

---

## 4. 개발환경 세팅

RAG / LangChain 실습을 위해 기존 TensorFlow 환경과 분리된 새로운 가상환경을 사용했다.

Python 버전:

```text
Python 3.11.16
```

예:

```bash
conda create -n langchain python=3.11 -y
conda activate langchain
```

필수 패키지 설치:

```bash
pip install langchain
pip install langchain_openai
pip install python-dotenv
```

VS Code에서 인터프리터가 바로 반영되지 않을 경우:

```text
Ctrl + Shift + P
        ↓
Developer: Reload Window
```

이후 Python Interpreter에서 `langchain (Python 3.11.x)` 환경을 선택한다.

---

## 5. 환경변수 설정

API Key는 코드에 직접 작성하지 않고 환경변수로 관리한다.

Windows에서는:

```text
제어판
 ↓
시스템
 ↓
고급 시스템 설정
 ↓
환경 변수
```

에서 사용자 변수 또는 시스템 변수를 추가할 수 있다.

예:

```text
변수 이름:
MONOROUTER_API_KEY

변수 값:
발급받은 API Key
```

API Key는 GitHub 등에 공개하면 안 된다.

---

## 6. .env 파일 사용

환경변수를 프로젝트 단위로 관리할 때 `.env` 파일을 사용할 수 있다.

```env
MONOROUTER_API_KEY=your_api_key
```

Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
```

`.env` 파일은 Git에 올라가지 않도록 `.gitignore`에 추가한다.

```gitignore
.env
```

---

## 7. API Key와 Base URL

오늘 사용한 모델 연결 방식에서는 API Key뿐 아니라 Base URL도 함께 지정했다.

```python
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"
```

구조:

```text
Python 코드
    ↓
API Key로 인증
    ↓
Base URL의 API 서버
    ↓
LLM
    ↓
응답
```

API Key는 인증에 사용되고, Base URL은 어느 API 서버로 요청을 보낼지 지정한다.

---

## 8. LangChain이란?

LangChain은 단순히 LLM 하나만 호출하는 것이 아니라, Prompt·Model·Parser·Retriever·Tool 등을 연결해서 AI 애플리케이션을 만들기 위한 프레임워크이다.

```text
LangChain
│
├─ Prompt
│   └─ 모델에게 어떤 질문을 할지 구성
│
├─ Model / LLM
│   └─ 실제 GPT 같은 언어모델
│
├─ Parser
│   └─ 모델 출력을 원하는 형태로 변환
│
├─ Retriever
│   └─ RAG에서 관련 문서 검색
│
├─ Memory
│   └─ 대화나 상태를 기억
│
└─ Tool
    └─ Agent가 실제 기능 실행
```

---

## 9. 휘발성 데이터와 Memory

현재 단순한 LLM 호출은 기본적으로 각 요청이 독립적이다.

```text
질문 1
↓
답변

질문 2
↓
질문 1의 내용을 자동으로 기억하지 않을 수 있음
```

따라서 이후에는 `Memory`, `Conversation History`, `State` 등을 이용해서 이전 대화 내용을 유지하는 구조를 배우게 된다.

---

## 10. PMO 구조

수업에서 정리한 기본 흐름:

```text
PMO

Prompt
 ↓
Model
 ↓
Output
```

즉 질문 형식을 만들고, 모델에 전달하고, 결과를 받는 구조이다.

---

## 11. PLP 구조

실제 LangChain에서 많이 사용하는 형태:

```text
PLP

Prompt
 ↓
LLM
 ↓
Parser
```

Parser를 추가하면 LLM의 출력을 프로그램에서 사용하기 쉬운 형태로 바꿀 수 있다.

```text
LLM
 ↓
AIMessage
 ↓
StrOutputParser
 ↓
문자열
```

---

## 12. LCEL

LCEL은 `LangChain Expression Language`이다.

가장 중요한 문법:

```python
|
```

예:

```python
chain = prompt | model
```

뜻:

```text
Prompt의 출력
      ↓
Model의 입력
```

Parser까지 연결하면:

```python
chain = prompt | model | output_parser
```

```text
Prompt
 ↓
Model
 ↓
Parser
 ↓
Output
```

---

## 13. 오늘 배운 코드

```python
# LCEL = LangChain Expression Language
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = PromptTemplate.from_template(
    "{topic}에 대해서 쉽게 {how} 설명해주세요."
)

model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

chain = prompt | model

input = {
    "topic": "양자컴퓨터 학습 원리",
    "how": "석사수준"
}

response = chain.invoke(input)

print(response.content)
```

---

## 14. 코드 흐름

### Prompt

```python
prompt = PromptTemplate.from_template(
    "{topic}에 대해서 쉽게 {how} 설명해주세요."
)
```

`{topic}`, `{how}`는 나중에 입력값이 들어가는 변수이다.

```python
input = {
    "topic": "양자컴퓨터 학습 원리",
    "how": "석사수준"
}
```

개념적으로 실제 Prompt는:

```text
양자컴퓨터 학습 원리에 대해서 쉽게 석사수준 설명해주세요.
```

가 된다.

### Model

```python
model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)
```

역할:

```text
Prompt
 ↓
LLM 호출
 ↓
AIMessage 반환
```

### Chain

```python
chain = prompt | model
```

```text
input
 ↓
PromptTemplate
 ↓
ChatOpenAI
 ↓
response
```

### invoke()

```python
response = chain.invoke(input)
```

의미:

> 이 Chain을 입력값으로 한 번 실행한다.

### response.content

`chain = prompt | model`까지만 연결했기 때문에 모델 결과는 메시지 객체 형태로 반환될 수 있다.

```python
print(response.content)
```

로 실제 답변 내용을 꺼낸다.

Parser를 추가하면:

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | model | StrOutputParser()

response = chain.invoke(input)

print(response)
```

처럼 바로 문자열 형태로 받을 수 있다.

---

## 15. LCEL 흐름 정리

현재 코드:

```python
chain = prompt | model
```

```text
input
 ↓
Prompt
 ↓
Model
 ↓
AIMessage
```

Parser 추가:

```python
chain = prompt | model | StrOutputParser()
```

```text
input
 ↓
Prompt
 ↓
Model
 ↓
Parser
 ↓
str
```

---

## 16. Prompt Engineering

최근 LLM의 성능이 강해지면서 예전처럼 복잡하고 긴 Prompt Engineering을 항상 해야 하는 경우는 줄어들고 있다.

하지만 Prompt 자체가 필요 없어졌다는 뜻은 아니다.

여전히 중요한 부분:

```text
역할 지정
출력 형식 지정
Context 제공
제약 조건 지정
RAG 검색 결과 전달
Tool 사용 조건
Agent 행동 규칙
```

즉 단순 질문에서는 Prompt Engineering의 부담이 줄었지만, RAG / Agent / Tool Calling에서는 Prompt 설계가 여전히 중요하다.

---

## 18. 기존 딥러닝 학습과 차이

지금까지:

```text
TensorFlow
Keras
DNN
CNN
RNN
LSTM
GRU
```

등을 배우며 모델이 데이터를 학습하는 원리를 익혔다.

기존 흐름:

```text
Data
 ↓
Model
 ↓
Loss
 ↓
Backpropagation
 ↓
Weight Update
```

이제 RAG부터는 AI 시스템을 연결하는 방향이 강해진다.

```text
Document
 ↓
Embedding
 ↓
Vector DB
 ↓
Retriever
 ↓
Prompt
 ↓
LLM
 ↓
Answer
```

그리고 Agent에서는:

```text
LLM
 ↓
판단
 ↓
Tool 선택
 ↓
Tool 실행
 ↓
결과 확인
 ↓
다음 행동
```

으로 발전한다.

---

