# 04. Furiosa RNGD NPU

Furiosa RNGD 기반 **LLM inference / serving / performance optimization** 실습을 정리합니다.

## Canonical Structure

```text
04_npu/
├── 00_environment/
├── 01_basics/
├── 02_langchain_tool/
├── 03_streaming_metrics/
├── 04_continuous_batching/
├── 05_kv_prefix_cache/
├── 06_concurrency_rps/
├── 07_server_metrics/
├── 08_model_porting/
└── 09_quantization/
```

## Learning Flow

```text
RNGD Environment
→ Basic Inference
→ LangChain / Tool Calling
→ Streaming / Latency / TTFC / TPS
→ Continuous Batching
→ KV Cache / Prefix Cache
→ Concurrency / RPS
→ Server Metrics
→ Model Porting
→ Quantization
```

## Environment

- RNGD
- Driver 2026.4.0
- Firmware 2026.4.0
- furiosa-llm 2026.4.0
- furiosa-llm-native 2026.4.0
- furiosa-tcc 2026.4.0

## PyTorch → NPU

SimpleModel에서:

- Linear(16 → 8)
- `torch.compile(model, backend=furiosa.torch.backend)`
- `furiosa:0`
- NPU inference 성공

ResNet18에서는 다음 이슈를 확인했습니다.

- binary compatibility
- unsupported ATen operator
- `aten::_native_batch_norm_legit_no_training`
- BatchNorm freeze / fold
- graph decomposition

## LLM Serving

주요 실습 모델:

- `furiosa-ai/Qwen3-8B-FP8`

주요 개념:

- OpenAI-compatible API
- streaming
- latency
- TTFC / TTFT
- prefill / decode
- TPS
- continuous batching
- KV cache
- prefix cache
- concurrency
- RPS
- server metrics

## Quantization

- BF16
- GGUF
- Q4_K_M
- llama.cpp
- Gemma / Qwen

## Current Status

GitHub에는 구조와 문서가 정리되어 있으며, NPU 서버의 `furiosa1/`, `furiosa2/` 실제 수업 코드를 위 구조로 동기화 중입니다.
