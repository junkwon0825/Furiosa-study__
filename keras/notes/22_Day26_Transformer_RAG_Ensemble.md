# Day 26 - Transformer Paper RAG / Ensemble Learning

## 1. 오늘 배운 핵심

```text
1. arXiv / AI Hub 같은 AI·논문 자료 출처
2. Transformer 논문 PDF를 이용한 RAG 챗봇
3. 학습과 추론의 관계
4. 앙상블 모델: Voting / Bagging / Boosting / Stacking
```

---

## 2. AI / 논문 자료 사이트

### arXiv

AI, 머신러닝, 딥러닝, NLP, Computer Vision, Robotics 등의 논문을 찾아볼 수 있는 대표적인 논문 사이트.

`Attention Is All You Need` 같은 Transformer 관련 논문도 arXiv에서 찾을 수 있다.

### AI Hub

국내 AI 학습용 데이터셋을 찾을 때 많이 사용하는 플랫폼.

```text
이미지 / 텍스트 / 음성 / 자율주행 / 산업 데이터 / 한국어 데이터
```

---

# 3. Transformer 논문 PDF 기반 챗봇

목표:

> `Attention Is All You Need` PDF를 Vector DB에 저장하고, 논문 내용을 기반으로 질문·요약·설명을 수행하는 RAG 챗봇 만들기

전체 구조:

```text
Attention Is All You Need.pdf
↓
PyPDFLoader
↓
Document
↓
Chunking
↓
Embedding
↓
Chroma
↓
Retriever
↓
Prompt
↓
LLM
↓
Gradio Chatbot
```

---

## 4. 파일 분리

```text
transformer_paper_save.py
PDF → Chunking → Embedding → Chroma 저장
```

```text
transformer_paper_chatbot.py
저장된 Chroma 불러오기 → Retriever → Prompt → LLM → Gradio
```

저장 코드와 챗봇 코드를 나누면 PDF를 실행할 때마다 다시 Embedding하지 않아도 된다.

---

# 7. Chroma 저장과 불러오기 차이

### 새 문서 저장

```python
Chroma.from_documents(...)
```

```text
Document
↓
Embedding
↓
Vector 생성
↓
Chroma 저장
```

### 기존 DB 사용

```python
Chroma(...)
```

```text
기존 Chroma DB
↓
연결
↓
Retriever / similarity search
```

즉:

```text
save.py    → Chroma.from_documents()
chatbot.py → Chroma()
```

---

# 8. RAG Pipeline

```text
사용자 질문
↓
Retriever
↓
Chroma
↓
관련 논문 Chunk 검색
↓
Relevant Documents
↓
Prompt
↓
LLM
↓
Answer
↓
Gradio
```

`Pipeline`은 여러 처리 단계를 순서대로 연결한 전체 흐름이다.

---

# 9. Retriever

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)
```

의미:

```text
질문
↓
Embedding
↓
Chroma
↓
가장 관련 있는 Chunk 4개 검색
```

`k=4`는 관련 문서를 4개 가져오겠다는 의미이다.

---

# 10. 학습과 추론

AI 모델의 추론 결과는 모델 구조만으로 결정되는 것이 아니라 학습 과정의 영향을 크게 받는다.

```text
Training
↓
Model
↓
Inference
```

추론 성능에 영향을 주는 요소:

```text
학습 데이터
모델 구조
Loss
Optimizer
전처리
Hyperparameter
학습 전략
```

즉 좋은 추론 결과를 얻기 위해서는 좋은 학습 과정이 중요하다.

---

# 11. 앙상블 모델

앙상블(Ensemble)은:

> 여러 모델의 예측을 합쳐 하나의 모델보다 더 안정적이고 정확한 예측을 만드는 방법

기본 구조:

```text
Model A ─┐
Model B ─┼→ 결과 결합 → 최종 예측
Model C ─┘
```

---

# 12. Voting

가장 단순한 앙상블 방식.

예:

```text
Logistic Regression
Random Forest
SVM
```

각 모델을 독립적으로 학습한 뒤 결과를 합친다.

## Hard Voting

분류 결과를 직접 다수결한다.

```text
Model A → 1
Model B → 0
Model C → 1
```

결과:

```text
1 = 2표
0 = 1표

최종 → 1
```

즉:

> 다수결

## Soft Voting

각 모델이 예측한 확률을 평균한다.

```text
Model A → 암 확률 0.9
Model B → 암 확률 0.6
Model C → 암 확률 0.8
```

평균:

```text
(0.9 + 0.6 + 0.8) / 3
= 약 0.766
```

최종적으로 약 76.6%의 확률을 기준으로 판단한다.

---

# 13. Bagging

Bagging = `Bootstrap Aggregating`

데이터를 여러 방식으로 랜덤 샘플링하여 여러 모델을 학습하고 결과를 합친다.

```text
Dataset
├→ Model 1
├→ Model 2
├→ Model 3
└→ Model 4
      ↓
   투표 / 평균
```

대표 모델:

```text
Random Forest
```

---

# 14. Boosting

이전 모델의 실수를 다음 모델이 계속 보완하는 방식.

```text
Model 1
↓
틀린 데이터 확인
↓
Model 2
↓
남은 오류 보완
↓
Model 3
↓
최종 결과
```

대표 모델:

```text
AdaBoost
Gradient Boosting
XGBoost
LightGBM
CatBoost
```

---

# 15. Stacking

여러 모델의 예측값을 다시 새로운 모델의 입력으로 사용한다.

```text
Model A ─┐
Model B ─┼→ Meta Model → Final Prediction
Model C ─┘
```

앞의 모델:

```text
Base Model
```

최종 결합 모델:

```text
Meta Model
```

즉 여러 모델의 결과를 어떻게 합칠지도 학습한다.

---

# 16. 앙상블 비교표

| 방법 | 모델 학습 방식 | 결과 결합 | 대표 모델 |
|---|---|---|---|
| Voting | 서로 독립적 | 투표 / 평균 | VotingClassifier |
| Bagging | 데이터 랜덤 샘플링 | 투표 / 평균 | RandomForest |
| Boosting | 이전 모델 오류 보완 | 가중 합 | XGBoost, LightGBM |
| Stacking | 여러 모델 독립 학습 | Meta Model | StackingClassifier |

---

# 17. 앙상블 핵심 정리

```text
Voting
= 여러 모델의 예측을 투표 또는 평균

Bagging
= 데이터를 다르게 뽑아 여러 모델 학습

Boosting
= 앞 모델의 실수를 다음 모델이 보완

Stacking
= 여러 모델 예측을 다시 다른 모델이 학습
```

