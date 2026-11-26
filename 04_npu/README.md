# Furiosa RNGD NPU Study

## Overview

This directory documents hands-on inference and serving exercises on a Furiosa RNGD NPU. The work progresses from basic `furiosa-llm` and OpenAI-compatible API calls to streaming metrics, Continuous Batching, KV/Prefix Cache behavior, concurrency, overlap scheduling, and LangChain Tool Calling/Agent workflows.

It also records a PyTorch model-porting exercise: a small model was compiled and executed on the NPU, while larger torchvision/ResNet attempts exposed backend, shared-library, and unsupported-operator constraints. No benchmark result or optimization gain is claimed beyond the experiments represented in the source files.

## Environment

The versions below come from [`00_environment/final_snapshot`](00_environment/final_snapshot/).

| Component | Recorded environment |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Accelerator | Furiosa RNGD (`npu0`) |
| Firmware | 2026.4.0 |
| Device driver | 2026.4.0 |
| Python | 3.11.18 |
| PyTorch / torchvision | 2.10.0 / 0.25.0 |
| Furiosa packages | `furiosa-llm`, `furiosa-llm-native`, `furiosa-kernels`, `furiosa-tcc` 2026.4.0 |
| `furiosa-smi` CLI | 2026.1.1 (the installed Python package is 2026.4.0) |
| LangChain | `langchain-core` 1.8.4, `langchain-openai` 1.7.4 |

The snapshot includes package inventories, NPU information, OS/kernel details, and the final server state. Model weights, caches, tokens, and other credentials are intentionally excluded.

## Learning Topics

| Topic | Description |
|---|---|
| Basic NPU Inference | Furiosa LLM, OpenAI-compatible API, direct FXB selection |
| Streaming & Metrics | Latency, TTFC/TTFT, TPS, token usage, prefill/decode timing |
| Continuous Batching | Concurrent request scheduling and worker-count experiments |
| KV / Prefix Cache | Cache reuse and server cache metrics |
| Concurrency / RPS | Single, sequential, and concurrent request patterns |
| Scheduling | Eager and overlap scheduling scripts; throughput/latency observation |
| Tool Calling | LangChain tools, tool-call inspection, and multi-step agents |
| Model Porting | `torch.compile`, backend selection, and unsupported ATen operators |
| Quantization | Notes on precision reduction, GGUF, and deployment preparation |

## Repository Structure

```text
04_npu/
├── 00_environment/              # Reproducibility and server snapshots
├── 01_basics/                   # Basic RNGD inference and FXB selection
├── 02_langchain_tool/           # LangChain connection and invoke/stream basics
├── 03_streaming_metrics/        # Latency, TTFC, TPS, and token metrics
├── 04_continuous_batching/      # Continuous Batching experiments
├── 05_kv_prefix_cache/          # KV Cache and Prefix Cache
├── 06_concurrency_rps/          # Single/sequential/concurrent requests and RPS
├── 07_server_metrics/           # Serving metrics endpoint inspection
├── 08_model_porting/            # PyTorch-to-NPU porting notes
├── 09_quantization/             # Quantization study notes
├── 10_scheduling_optimization/  # Eager vs overlap scheduling
└── 11_tool_calling/             # Tool Calling and Agent progression
```

## Key Experiments

1. **Basic RNGD inference**  
   Sent requests through Furiosa LLM and an OpenAI-compatible client, including a direct FXB selection example.

2. **Latency, TPS, and TTFC**  
   Measured request latency and token generation timing, then inspected streaming chunks, usage metadata, and prefill/decode phases.

3. **Continuous Batching**  
   Issued concurrent workloads while varying the request/worker setup to observe total processing time and per-request latency.

4. **KV Cache and Prefix Cache**  
   Repeated prompts and queried cache-related metrics to study cache reuse and hit behavior.

5. **Sequential vs concurrent requests**  
   Compared single, sequential, and concurrent request structures and calculated RPS from observed completion timing.

6. **Overlap scheduling**  
   Used eager and overlap-oriented server configurations to examine the scheduling trade-off between CPU preparation and NPU execution.

7. **LangChain Tool Calling and Agent**  
   Progressed from tool schemas and tool-call inspection to multiple tools, multi-step agents, recursion limits, and an interactive CLI.

8. **PyTorch model porting**  
   Compiled a small `SimpleModel` with the Furiosa backend and documented compatibility failures encountered with more complex models.

## Agent & Tool Calling

The later exercises connect a LangChain Agent to an LLM served on RNGD:

```text
User Request
    ↓
RNGD-served LLM
    ↓
LangChain Agent
    ↓
Tool Selection
    ↓
Python Tool Execution
    ↓
Tool Result
    ↓
Final Response
```

The implemented arithmetic tools are `add`, `subtract`, `multiply`, and `divide`. The progression in [`11_tool_calling`](11_tool_calling/) covers tool-call argument inspection, multiple tools, multi-step calculations, `recursion_limit`, error handling, and a loop-based CLI Agent. Its local `kingkeras.py` preserves the serving configuration used for automatic tool selection and the Hermes tool-call parser.

## Model Porting / Troubleshooting

| Observation | Likely cause or context | Lesson |
|---|---|---|
| `SimpleModel` inference succeeded with `torch.compile(model, backend=furiosa.torch.backend)` | The graph used supported operations and the correct backend object | Start with a minimal graph and verify the backend path first |
| `InvalidBackend` with an incorrect backend name | Backend registration/name mismatch | Validate the backend API before debugging model code |
| Shared-library compatibility errors in torchvision/ResNet attempts | Runtime and binary-version compatibility | Check the Python, PyTorch, torchvision, and NPU runtime matrix together |
| `aten::_native_batch_norm_legit_no_training` was unsupported | The compiled graph contained an operator unavailable to the backend | Inspect the lowered graph and consider freeze/fold/decomposition options |

These failures are retained because model porting depends not only on model architecture, but also on graph lowering, operator coverage, and binary compatibility.

## What I Learned

- NPU inference work extends beyond calling a model: serving configuration, token timing, batching, cache behavior, and scheduling all affect how a workload is observed.
- Latency, TTFC/TTFT, TPS, and RPS describe different parts of serving behavior and should be measured separately.
- Concurrency experiments require both aggregate throughput and per-request latency to understand the trade-off.
- Tool Calling becomes an Agent workflow only after tool selection, execution, result feedback, multi-step control, and error handling are connected.
- PyTorch-to-NPU porting requires early validation of backend selection, shared-library compatibility, graph structure, and operator support.

Overall learning flow:

```text
NPU Inference
→ Serving
→ Performance Analysis
→ Scheduling / Cache
→ Tool Calling / Agent
→ Model Porting
```
