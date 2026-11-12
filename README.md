# Furiosa AI Agent School Study

> Python · Machine Learning · Deep Learning · RAG · AI Agent · Furiosa RNGD NPU

2026년 Furiosa AI Agent School에서 학습한 코드와 실습 기록을 정리하는 저장소입니다.  
초기에는 수업 코드를 날짜 순으로 저장했지만, 2026-11-12부터 **기술 영역 + NPU 실습 중심**으로 구조를 정리하고 있습니다.

## Course Progress

| Item | Status |
|---|---|
| Start | 2026-08-31 |
| Current | 2026-11-12 |
| Progress | Day 54 / 80 (평일 기준) |
| Main Focus | AI Model → Compile → NPU Runtime → Inference → Serving |

## Learning Map

```text
Python / NumPy
      ↓
Machine Learning
      ↓
TensorFlow / Keras
      ↓
CNN / RNN / Transformer
      ↓
Embedding / Vector DB / RAG
      ↓
AI Agent / MCP
      ↓
Furiosa RNGD NPU
      ↓
Compile / Quantization / Inference / Serving
```

## Repository Guide

| Path | Contents |
|---|---|
| `01_foundations/` | Python, sklearn, CV, XGBoost, hyperparameter tuning |
| `02_deep_learning/` | TensorFlow/Keras, CNN, RNN, Transformer |
| `03_retrieval_agent/` | Embedding, FAISS, Chroma, RAG, Agent, MCP |
| `04_npu/` | Furiosa RNGD, torch.compile, model porting, LLM serving |
| `docs/` | 학습 로그, 오류/해결 기록 |
| `python/`, `ml/`, `keras/`, `keras2/`, `RAG/`, `LG/`, `논문/` | 기존 수업 원본 코드/자료 (보존) |

> 기존 폴더는 수업 당시의 코드 흐름을 보존하기 위해 당분간 유지합니다. 신규 정리본과 NPU 실습은 번호가 붙은 폴더에 추가합니다.

## Current NPU Track

### 1. RNGD Environment

- `furiosa-smi info`, `furiosa-smi version`
- driver / firmware / runtime 확인
- Hugging Face cache와 개인 token 분리

### 2. PyTorch → NPU

```text
PyTorch Model
   ↓
torch.compile(..., backend=furiosa.torch.backend)
   ↓
Furiosa Runtime
   ↓
RNGD
   ↓
NPU Inference
```

- Linear(16 → 8) SimpleModel NPU inference 성공
- `furiosa:0` device 확인
- ResNet18 포팅 중 unsupported ATen / BatchNorm 문제 분석

### 3. Quantization

- BF16 / GGUF
- Q4_K_M
- llama.cpp
- Gemma / Qwen 실습

### 4. LLM Serving

```text
LangChain
   ↓
OpenAI-compatible API
   ↓
furiosa-llm
   ↓
Continuous Batching
   ↓
Qwen3
   ↓
RNGD
```

- Qwen3-8B-FP8
- Continuous batching
- KV cache
- latency / throughput / TTFT / TPS

## Recent ML Track

- KFold / StratifiedKFold
- cross_val_score / cross_val_predict
- all_estimators
- XGBoost
- GridSearchCV
- RandomizedSearchCV
- HalvingGridSearchCV
- KerasRegressor + RandomizedSearchCV
- EarlyStopping / ReduceLROnPlateau

## Current Goal

단순 모델 학습에서 끝내지 않고 다음 전체 inference stack을 이해하는 것이 목표입니다.

```text
Model
→ Graph / Operator
→ Compile
→ Quantization
→ Accelerator Runtime
→ NPU Inference
→ Serving
→ Benchmark
→ Optimization
```

## Next

- NPU 서버의 `furiosa1/`, `furiosa2/` 실제 수업 코드 동기화
- SimpleModel / ResNet18 / YOLO 포팅 기록 추가
- Furiosa LLM serving 코드 추가
- GPU vs NPU latency / throughput benchmark
- troubleshooting 문서 지속 업데이트

_Last updated: 2026-11-12_
