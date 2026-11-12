# 04. Furiosa RNGD NPU

이 저장소에서 가장 중요한 NPU 실습 영역입니다.

## Environment

확인한 주요 환경:
- RNGD
- Driver 2026.4.0
- Firmware 2026.4.0
- `furiosa-llm 2026.4.0`
- `furiosa-llm-native 2026.4.0`
- `furiosa-tcc 2026.4.0`
- PyTorch + Furiosa backend

## 1. PyTorch Compile

SimpleModel:
- Linear(16 → 8)
- `torch.compile(model, backend=furiosa.torch.backend)`
- NPU inference 성공
- output shape: `[1, 8]`

## 2. Model Porting

ResNet18에서 확인한 문제:
- shared-library / torch compatibility
- unsupported ATen operator
- `aten::_native_batch_norm_legit_no_training`
- BatchNorm freeze / graph decomposition 검토

## 3. Quantization

- BF16
- GGUF
- Q4_K_M
- llama.cpp
- Gemma / Qwen

## 4. LLM Serving

실습 대상:
- `furiosa-ai/Qwen3-8B-FP8`

예시:

```bash
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:0 \
    --host 127.0.0.1 \
    --port 8000
```

확인:

```bash
curl http://127.0.0.1:8000/v1/models
```

학습 포인트:
- OpenAI-compatible API
- continuous batching
- KV cache
- TTFT
- TPS
- latency / throughput

## 5. Server Sync

NPU 서버에서 실제 실습 코드를 이 폴더 아래로 동기화할 예정입니다.

권장 구조:

```text
04_npu/
├── 01_environment/
├── 02_torch_compile/
├── 03_model_porting/
├── 04_quantization/
├── 05_llm_serving/
└── 06_benchmark/
```
