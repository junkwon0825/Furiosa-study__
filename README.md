# Furiosa AI Agent School 학습 기록

> Python과 데이터 처리 → 머신러닝 → 딥러닝 → LLM/RAG → AI Agent → Furiosa RNGD NPU

## 개요

이 저장소는 **Furiosa AI Agent School**에서 직접 수행한 수업 코드와 실습 기록을 정리한 공간입니다. Python과 데이터 전처리부터 머신러닝, TensorFlow/Keras 딥러닝, 검색 증강 생성(RAG), AI Agent workflow, Furiosa RNGD NPU 추론까지 단계적으로 학습했습니다.

교육과정의 진행 순서는 유지하면서 코드를 기술 주제별로 재구성했습니다. README의 내용은 저장소에서 실제 구현이나 트러블슈팅 기록으로 확인할 수 있는 범위만 포함하며, 측정하지 않은 benchmark 수치나 성능 개선 결과는 작성하지 않았습니다.

## 학습 여정

```text
Python / 데이터 처리
  함수, module, Pandas, 결측치 전처리
      ↓
머신러닝
  교차검증, 모델 비교, XGBoost 탐색
      ↓
딥러닝
  Dense, CNN, RNN/LSTM, callback, TensorBoard
      ↓
LLM / RAG
  chunking, embedding, Chroma/FAISS, BM25, retrieval chain
      ↓
AI Agent
  LangGraph state, ToolNode, memory, 다단계 도구 실행
      ↓
Furiosa RNGD NPU
  추론, 서빙, 성능 지표, cache, scheduling, 모델 포팅
```

## 교육과정과 학습 주제

| 영역 | 직접 실습한 내용 | 코드 |
|---|---|---|
| 기초 | Python 함수, lambda, package, Pandas indexing, interpolation, imputation | [`01_foundations`](01_foundations/) |
| 머신러닝 | Feature importance, KFold/StratifiedKFold, estimator 비교, Grid/Random/Halving Search, XGBoost | [`02_machine_learning`](02_machine_learning/) |
| 딥러닝 | Dense network, 회귀/분류, CNN, augmentation, RNN/LSTM/Conv1D, callback, TensorBoard | [`03_deep_learning`](03_deep_learning/) |
| LLM / RAG | LCEL, embedding, Chroma, FAISS, PDF retrieval, BM25/vector hybrid retrieval, Gradio | [`04_rag_llm`](04_rag_llm/) |
| AI Agent | LangGraph `StateGraph`, `ToolNode`, conditional edge, memory, parallel branch | [`05_ai_agent`](05_ai_agent/) |
| NPU / Edge AI | RNGD 추론과 서빙, latency/TPS/TTFC/RPS, batching, cache, scheduling, Agent, 모델 포팅 | [`06_npu`](06_npu/) |

## 저장소 구조

```text
Furiosa-study__/
├── 01_foundations/        # Python, package, Pandas 전처리
├── 02_machine_learning/   # scikit-learn과 XGBoost 실습
├── 03_deep_learning/      # TensorFlow/Keras 학습 과정
├── 04_rag_llm/            # 검색과 RAG pipeline
├── 05_ai_agent/           # LangGraph Agent 실습
├── 06_npu/                # Furiosa RNGD NPU 추론과 서빙
├── docs/                  # 학습 여정, 참고자료, 트러블슈팅
├── README.md
└── .gitignore
```

## 주요 기술 실습

- Pandas interpolation과 scikit-learn imputer를 사용해 결측치 처리 예제를 구현했습니다.
- 교차검증으로 classifier/regressor 후보를 비교하고, GridSearchCV·RandomizedSearchCV·HalvingGridSearchCV로 XGBoost hyperparameter를 탐색했습니다.
- TensorFlow/Keras로 Dense, CNN, RNN/LSTM, Conv1D 모델을 구성하고 validation, EarlyStopping, ModelCheckpoint, learning-rate 조절, TensorBoard 기록을 실습했습니다.
- 문서 loading, chunking, embedding, Chroma/FAISS vector search, retriever, LLM 응답 생성으로 RAG pipeline을 구현했습니다.
- BM25와 vector retrieval을 결합한 hybrid search를 실습했습니다.
- LangGraph의 state, conditional routing, `ToolNode`, checkpoint memory, parallel branch로 Agent workflow를 구성했습니다.
- LangChain 도구와 다단계 Agent를 Furiosa RNGD에서 서빙하는 LLM에 연결했습니다.
- Latency, TTFC, TPS, RPS, token metadata, Continuous Batching, KV/Prefix Cache, overlap scheduling을 통해 LLM 서빙 동작을 측정하고 관찰했습니다.
- 작은 PyTorch 모델의 RNGD 포팅 성공 과정과 미지원 연산자·binary 호환성 실패 사례를 엔지니어링 기록으로 보존했습니다.

## Furiosa RNGD NPU

NPU 실습은 전체 학습 과정의 마지막 심화 단계입니다.

```text
RNGD 기본 추론
→ OpenAI 호환 서빙
→ Streaming과 Token 지표
→ Continuous Batching
→ KV / Prefix Cache
→ Concurrency / RPS
→ Overlap Scheduling
→ Tool Calling / Agent
→ PyTorch 모델 포팅
```

저장된 환경정보에는 Ubuntu 22.04.5, Python 3.11.18, Furiosa package/driver/firmware 2026.4.0, PyTorch 2.10.0, LangChain 패키지 버전이 기록되어 있습니다. 자세한 내용은 [`06_npu/README.md`](06_npu/README.md)와 환경 snapshot에서 확인할 수 있습니다.

## AI Agent

이 저장소에서는 다음 세 단계를 구분해 학습했습니다.

```text
LLM          → text 생성
Tool Calling → LLM이 실행할 함수와 인자를 선택
Agent        → LLM + 도구 + 반복 실행 + 상태 및 제어 흐름
```

LangGraph 실습에서는 graph state, node/edge, 조건부 tool routing, memory, 병렬 실행을 다뤘습니다. RNGD Agent 실습에서는 산술 도구, tool call 확인, 여러 도구의 선택, `recursion_limit`, 오류 처리, 대화형 CLI 흐름을 추가로 구현했습니다.

## 트러블슈팅과 엔지니어링 학습

구현 과정에서 의사결정에 영향을 준 실패 사례도 함께 보존했습니다.

- 분류의 accuracy는 회귀 평가 지표가 아니므로 문제 유형에 맞는 metric을 선택해야 합니다. 후속 회귀 예제에서는 model score와 R² 중심으로 평가 흐름을 정리했습니다.
- Data leakage를 방지하려면 scaling과 전처리 객체를 training data에만 `fit`해야 합니다.
- RAG 실습을 document loading, chunking, embedding, vector store, retrieval, answer generation 단계로 분리하면 오류 구간을 확인하기 쉽습니다.
- NPU 모델 포팅은 backend 등록, PyTorch/torchvision binary 호환성, graph lowering, 지원 연산자의 영향을 함께 받습니다.
- ResNet 계열 포팅 과정에서 `aten::_native_batch_norm_legit_no_training` 미지원 오류를 경험했습니다.

자세한 내용은 [`docs/troubleshooting.md`](docs/troubleshooting.md)에 기록했습니다.

## 기술 스택

`Python` · `NumPy` · `Pandas` · `scikit-learn` · `XGBoost` · `TensorFlow/Keras` · `PyTorch` · `LangChain` · `LangGraph` · `Chroma` · `FAISS` · `Gradio` · `Furiosa SDK` · `Furiosa RNGD` · `Git/GitHub`

## 참고사항

- 대용량 dataset, 모델 weight, vector DB, cache, 인증정보, 생성된 학습 artifact는 Git에 포함하지 않았습니다.
- 저장소의 코드는 수업에서 진행한 실습 흐름을 중심으로 보존했습니다. 일부 파일은 중간 학습 단계를 의도적으로 남겨두었으며, 후속 예제와 각 주제 README에서 수정되거나 완성된 흐름을 안내합니다.
