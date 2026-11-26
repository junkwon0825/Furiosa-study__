# 기술 주제별 학습 여정

기존 날짜별 강의 노트를 코드와 연결되는 기술 주제 중심으로 통합한 문서입니다. 상세 구현은 각 영역의 README와 실제 Python 파일에서 확인할 수 있습니다.

## Python과 데이터 처리

- 함수, lambda, module/package import
- Pandas `loc`/`iloc`, 결측치 확인과 보간
- sklearn imputer를 이용한 전처리

## Machine Learning

- 분류와 회귀의 문제 유형 및 평가 방식 구분
- Scaling과 train/test data leakage 방지
- KFold/StratifiedKFold 교차검증
- Tree/XGBoost feature importance
- Grid, Randomized, Halving hyperparameter search
- Bagging, Voting, Stacking의 관계는 수업에서 개념으로 학습했으며, 저장소의 독립 sklearn 구현으로 과장하지 않음

## Deep Learning

- Dense network의 input/hidden/output 구조
- Train/validation/test 분리와 loss/metric 관찰
- EarlyStopping, ModelCheckpoint, Dropout, ReduceLROnPlateau
- CNN의 convolution, padding, stride, pooling, augmentation
- RNN/LSTM/GRU와 time-series window 구성
- Tokenizer, padding, embedding, Conv1D
- Multi-input/output model과 transfer learning
- TensorBoard 기반 학습 로그 확인

## LLM과 RAG

- LangChain prompt/model/output parser 흐름
- Document loading, chunking, embedding
- Chroma와 FAISS 기반 vector search
- Retriever와 retrieval chain
- PDF RAG, Gradio chatbot
- BM25와 vector retrieval을 결합한 hybrid search

## AI Agent

- LangGraph state와 message reducer
- Node/edge/conditional routing
- ToolNode와 검색 도구 실행
- Checkpoint memory
- Parallel branch와 결과 merge
- RNGD-served LLM 기반 multi-tool Agent

## Furiosa RNGD NPU

- RNGD 환경 및 driver/firmware 확인
- OpenAI-compatible LLM serving
- Latency, TTFC, TPS, RPS와 token metadata
- Continuous Batching, KV/Prefix Cache, concurrency
- Overlap scheduling
- Tool Calling과 interactive Agent
- PyTorch model porting 및 unsupported operator troubleshooting
