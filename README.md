# Furiosa AI Agent School Study

> Python & Data → Machine Learning → Deep Learning → LLM/RAG → AI Agent → Furiosa RNGD NPU

## Overview

This repository documents hands-on coursework from the **Furiosa AI Agent School**. The curriculum progressed from Python and data preprocessing to machine learning, TensorFlow/Keras, retrieval-augmented generation, AI Agent workflows, and inference on the Furiosa RNGD NPU.

The repository keeps the learning sequence visible while organizing the code by technical topic. Claims in this README are limited to implementations and troubleshooting records that are present in the repository; generated benchmark numbers are not added.

## Learning Journey

```text
Python / Data
  functions, modules, Pandas, missing-value preprocessing
      ↓
Machine Learning
  cross-validation, model comparison, XGBoost search
      ↓
Deep Learning
  Dense, CNN, RNN/LSTM, callbacks, TensorBoard
      ↓
LLM / RAG
  chunking, embedding, Chroma/FAISS, BM25, retrieval chain
      ↓
AI Agent
  LangGraph state, ToolNode, memory, multi-step tool execution
      ↓
Furiosa RNGD NPU
  inference, serving, metrics, cache, scheduling, model porting
```

## Curriculum & Topics

| Area | Hands-on topics | Evidence |
|---|---|---|
| Foundations | Python functions, lambda, packages, Pandas indexing, interpolation, imputation | [`01_foundations`](01_foundations/) |
| Machine Learning | Feature importance, KFold/StratifiedKFold, estimator comparison, Grid/Random/Halving search, XGBoost | [`02_machine_learning`](02_machine_learning/) |
| Deep Learning | Dense networks, regression/classification, CNN, augmentation, RNN/LSTM/Conv1D, callbacks, TensorBoard | [`03_deep_learning`](03_deep_learning/) |
| LLM / RAG | LCEL, embedding, Chroma, FAISS, PDF retrieval, BM25/vector hybrid retrieval, Gradio | [`04_rag_llm`](04_rag_llm/) |
| AI Agent | LangGraph `StateGraph`, `ToolNode`, conditional edges, memory, parallel branches | [`05_ai_agent`](05_ai_agent/) |
| NPU / Edge AI | RNGD inference and serving, latency/TPS/TTFC/RPS, batching, cache, scheduling, Agent, model porting | [`06_npu`](06_npu/) |

## Repository Structure

```text
Furiosa-study__/
├── 01_foundations/        # Python, packages, Pandas preprocessing
├── 02_machine_learning/   # sklearn and XGBoost experiments
├── 03_deep_learning/      # TensorFlow/Keras learning sequence
├── 04_rag_llm/            # Retrieval and RAG pipelines
├── 05_ai_agent/           # LangGraph Agent exercises
├── 06_npu/                # Furiosa RNGD NPU inference and serving
├── docs/                  # Learning journey, references, troubleshooting
├── README.md
└── .gitignore
```

## Key Technical Highlights

- Applied Pandas interpolation and sklearn imputers to missing-value preprocessing examples.
- Compared classifier/regressor families with cross-validation and searched XGBoost hyperparameters using GridSearchCV, RandomizedSearchCV, and HalvingGridSearchCV.
- Built TensorFlow/Keras Dense, CNN, RNN/LSTM, and Conv1D models with validation, EarlyStopping, ModelCheckpoint, learning-rate control, and TensorBoard logging.
- Implemented RAG pipelines using document loading, chunking, embeddings, Chroma/FAISS vector search, retrievers, and LLM response generation.
- Combined BM25 and vector retrieval in a hybrid-search exercise.
- Built LangGraph flows with state, conditional routing, `ToolNode`, checkpoint memory, and parallel branches.
- Connected LangChain tools and multi-step Agents to an LLM served on Furiosa RNGD.
- Measured LLM serving behavior through latency, TTFC, TPS, RPS, token metadata, Continuous Batching, KV/Prefix Cache, and overlap-scheduling exercises.
- Recorded PyTorch-to-RNGD porting success for a small model and retained unsupported-operator and binary-compatibility failures as engineering evidence.

## Furiosa RNGD NPU

The NPU track is the final stage of the learning progression:

```text
Basic RNGD Inference
→ OpenAI-compatible Serving
→ Streaming and Token Metrics
→ Continuous Batching
→ KV / Prefix Cache
→ Concurrency / RPS
→ Overlap Scheduling
→ Tool Calling / Agent
→ PyTorch Model Porting
```

The recorded environment includes Ubuntu 22.04.5, Python 3.11.18, Furiosa packages/driver/firmware 2026.4.0, PyTorch 2.10.0, and LangChain package versions. See [`06_npu/README.md`](06_npu/README.md) and the saved environment snapshot for details.

## AI Agent

The repository distinguishes three levels of behavior:

```text
LLM          → text generation
Tool Calling → the LLM selects a callable function and arguments
Agent        → LLM + tools + repeated execution + state/control flow
```

The LangGraph exercises cover graph state, nodes/edges, conditional tool routing, memory, and parallel execution. The RNGD Agent exercises add arithmetic tools, tool-call inspection, multiple tools, `recursion_limit`, error handling, and an interactive CLI flow.

## Troubleshooting & Engineering Lessons

The repository keeps failures that affected implementation decisions:

- Correct metric selection matters: classification accuracy is not a regression metric; regression examples use model score/R²-oriented evaluation in the later corrected flow.
- Scaling and preprocessing must be fitted on training data to avoid leakage.
- RAG experiments separate document loading, chunking, embedding, vector storage, retrieval, and answer generation so failures can be isolated.
- NPU model porting depends on backend registration, PyTorch/torchvision binary compatibility, graph lowering, and supported operators.
- `aten::_native_batch_norm_legit_no_training` was encountered as an unsupported operator during a ResNet-family porting attempt.

More detail is available in [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Tech Stack

`Python` · `NumPy` · `Pandas` · `scikit-learn` · `XGBoost` · `TensorFlow/Keras` · `PyTorch` · `LangChain` · `LangGraph` · `Chroma` · `FAISS` · `Gradio` · `Furiosa SDK` · `Furiosa RNGD` · `Git/GitHub`

## Notes

- Large datasets, model weights, vector databases, caches, credentials, and generated training artifacts are excluded from Git.
- The code primarily preserves classroom experiments. Some files intentionally show intermediate learning stages; later examples and topic READMEs identify the corrected or more complete flow.
