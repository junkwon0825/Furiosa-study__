# NPU Server Sync Guide

NPU 서버의 실제 수업 코드를 GitHub `04_npu/` 구조에 동기화합니다.

## Source

```text
/root/furiosa_study/furiosa1/
/root/furiosa_study/furiosa2/
```

## Destination

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

## Mapping

- f01, f02, f04, f11 → `01_basics/`
- tool calling / LangChain / pkg invoke → `02_langchain_tool/`
- TPS / stream / latency / TTFC / prefill / decode → `03_streaming_metrics/`
- f21 continuous batching → `04_continuous_batching/`
- f22 KV cache / f23 prefix cache → `05_kv_prefix_cache/`
- f24~f28 request / concurrency / RPS → `06_concurrency_rps/`
- f29 metrics → `07_server_metrics/`

## Before Push

```bash
git status
grep -RniE "api[_-]?key|token|password|secret|hf_[A-Za-z0-9]+" 04_npu || true
```

Do not commit:

- API keys / tokens
- .env
- .pem / .key
- GGUF / FXB / model weights
- Hugging Face cache
- vector DB cache

## Push

```bash
git add 04_npu
git commit -m "add Furiosa NPU inference and serving study codes"
git push origin main
```
