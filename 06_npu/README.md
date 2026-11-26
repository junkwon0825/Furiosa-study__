# Furiosa RNGD NPU 학습

## 개요

이 폴더는 Furiosa RNGD NPU에서 직접 진행한 추론 및 서빙 실습을 정리한 공간입니다. 기본적인 `furiosa-llm` 사용과 OpenAI 호환 API 호출부터 스트리밍 지표, Continuous Batching, KV/Prefix Cache 동작, 동시성 처리, overlap scheduling, LangChain Tool Calling 및 Agent 구현까지 단계적으로 학습했습니다.

PyTorch 모델 포팅 과정도 함께 기록했습니다. 작은 모델은 NPU용으로 컴파일하여 실행했고, torchvision/ResNet 계열 모델을 시도하면서 backend 설정, shared library 호환성, 미지원 연산자 문제를 경험했습니다. 소스 코드로 확인할 수 있는 실험 범위만 기술했으며, 측정하지 않은 성능 결과나 최적화 효과는 포함하지 않았습니다.

## 실행 환경

아래 버전은 [`00_environment/final_snapshot`](00_environment/final_snapshot/)에 저장된 실제 환경정보를 기준으로 작성했습니다.

| 구성 요소 | 기록된 환경 |
|---|---|
| 운영체제 | Ubuntu 22.04.5 LTS |
| 가속기 | Furiosa RNGD (`npu0`) |
| 펌웨어 | 2026.4.0 |
| 디바이스 드라이버 | 2026.4.0 |
| Python | 3.11.18 |
| PyTorch / torchvision | 2.10.0 / 0.25.0 |
| Furiosa 패키지 | `furiosa-llm`, `furiosa-llm-native`, `furiosa-kernels`, `furiosa-tcc` 2026.4.0 |
| `furiosa-smi` CLI | 2026.1.1 (설치된 Python 패키지는 2026.4.0) |
| LangChain | `langchain-core` 1.8.4, `langchain-openai` 1.7.4 |

환경 snapshot에는 패키지 목록, NPU 정보, 운영체제와 커널 정보, 서버의 최종 상태가 포함되어 있습니다. 모델 weight, cache, token 및 기타 인증정보는 의도적으로 제외했습니다.

## 학습 주제

| 주제 | 내용 |
|---|---|
| 기본 NPU 추론 | Furiosa LLM, OpenAI 호환 API, FXB 직접 지정 |
| 스트리밍과 성능 지표 | Latency, TTFC/TTFT, TPS, token usage, prefill/decode 시간 |
| Continuous Batching | 동시 요청 스케줄링과 worker 수 변화 실험 |
| KV / Prefix Cache | Cache 재사용과 서버 cache 지표 확인 |
| 동시성 / RPS | 단일·순차·동시 요청 구조 비교 |
| 스케줄링 | Eager/overlap scheduling과 처리량·지연시간 관찰 |
| Tool Calling | LangChain 도구, tool call 분석, 다단계 Agent |
| 모델 포팅 | `torch.compile`, backend 선택, 미지원 ATen 연산자 |
| 양자화 | 정밀도 축소, GGUF, NPU 배포 준비 학습 |

## 저장소 구조

```text
04_npu/
├── 00_environment/              # 재현을 위한 환경 및 서버 snapshot
├── 01_basics/                   # RNGD 기본 추론과 FXB 지정
├── 02_langchain_tool/           # LangChain 연결과 invoke/stream 기초
├── 03_streaming_metrics/        # Latency, TTFC, TPS, token 지표
├── 04_continuous_batching/      # Continuous Batching 실험
├── 05_kv_prefix_cache/          # KV Cache와 Prefix Cache
├── 06_concurrency_rps/          # 단일·순차·동시 요청과 RPS
├── 07_server_metrics/           # 서빙 metrics endpoint 확인
├── 08_model_porting/            # PyTorch 모델의 NPU 포팅 기록
├── 09_quantization/             # 양자화 학습 기록
├── 10_scheduling_optimization/  # Eager와 overlap scheduling 비교
└── 11_tool_calling/             # Tool Calling과 Agent 구현 과정
```

## 주요 실험

1. **RNGD 기본 추론**
   Furiosa LLM과 OpenAI 호환 클라이언트로 요청을 전송하고, FXB를 직접 지정하는 방식도 확인했습니다.

2. **Latency, TPS, TTFC 측정**
   요청 지연시간과 token 생성 시간을 측정하고, streaming chunk, usage metadata, prefill/decode 구간을 확인했습니다.

3. **Continuous Batching**  
   요청 및 worker 구성을 바꾸면서 동시 요청을 보내고, 전체 처리시간과 요청별 지연시간을 관찰했습니다.

4. **KV Cache와 Prefix Cache**
   동일하거나 유사한 prompt를 반복 요청하고 cache 관련 지표를 조회하여 재사용과 hit 동작을 학습했습니다.

5. **순차 요청과 동시 요청 비교**
   단일·순차·동시 요청 코드를 각각 작성하고, 실제 완료시간을 바탕으로 RPS를 계산했습니다.

6. **Overlap scheduling**  
   Eager 방식과 overlap 중심의 서버 설정을 사용해 CPU 준비 작업과 NPU 실행을 겹치는 스케줄링 구조를 확인했습니다.

7. **LangChain Tool Calling과 Agent**
   Tool schema와 tool call 분석부터 여러 도구의 실행, 다단계 Agent, recursion limit, 대화형 CLI까지 순서대로 구현했습니다.

8. **PyTorch 모델 포팅**
   작은 `SimpleModel`을 Furiosa backend로 컴파일했으며, 더 복잡한 모델에서 발생한 호환성 문제도 함께 기록했습니다.

## 에이전트와 도구 호출

후반부 실습에서는 RNGD에서 서빙하는 LLM과 LangChain Agent를 다음 흐름으로 연결했습니다.

```text
사용자 요청
    ↓
RNGD에서 서빙하는 LLM
    ↓
LangChain Agent
    ↓
도구 선택
    ↓
Python 도구 실행
    ↓
도구 실행 결과
    ↓
최종 응답
```

구현한 산술 도구는 `add`, `subtract`, `multiply`, `divide`입니다. [`11_tool_calling`](11_tool_calling/)에서는 tool call 인자 확인, 여러 도구의 선택과 실행, 다단계 계산, `recursion_limit`, 오류 처리, 반복 입력을 받는 CLI Agent까지 구현했습니다. 해당 폴더의 `kingkeras.py`에는 자동 도구 선택과 Hermes tool call parser를 사용한 서빙 설정이 기록되어 있습니다.

## 모델 포팅과 문제 해결

| 관찰한 문제 또는 결과 | 원인 후보와 상황 | 배운 점 |
|---|---|---|
| `torch.compile(model, backend=furiosa.torch.backend)`로 `SimpleModel` 추론 성공 | 지원되는 연산과 올바른 backend 객체를 사용한 작은 graph | 최소 모델로 backend 경로부터 검증해야 함 |
| 잘못된 backend 이름에서 `InvalidBackend` 발생 | Backend 등록 이름 또는 호출 방식 불일치 | 모델 코드를 점검하기 전에 backend API를 확인해야 함 |
| torchvision/ResNet 시도 중 shared library 호환성 오류 | Runtime과 binary 버전 조합 문제 | Python, PyTorch, torchvision, NPU runtime 버전을 함께 확인해야 함 |
| `aten::_native_batch_norm_legit_no_training` 미지원 | 컴파일 graph에 backend가 지원하지 않는 연산 포함 | Lowered graph를 확인하고 freeze, fold, decomposition 가능성을 검토해야 함 |

모델 포팅은 모델 구조만의 문제가 아니라 graph lowering, 연산자 지원 범위, binary 호환성의 영향을 함께 받는다는 점을 확인했습니다. 이러한 이유로 성공 사례뿐 아니라 실패 과정도 삭제하지 않고 기록했습니다.

## 배운 점

- NPU 추론은 모델 호출만으로 끝나지 않으며, 서빙 설정, token 생성 시간, batching, cache, scheduling을 함께 이해해야 합니다.
- Latency, TTFC/TTFT, TPS, RPS는 서로 다른 구간과 특성을 나타내므로 각각 구분해 측정해야 합니다.
- 동시성 실험에서는 전체 처리량과 개별 요청의 지연시간을 함께 확인해야 trade-off를 이해할 수 있습니다.
- Tool Calling은 도구 선택, 실행, 결과 전달, 다단계 제어, 오류 처리가 연결되어야 실제 Agent 흐름이 됩니다.
- PyTorch 모델을 NPU로 포팅할 때는 backend 선택, shared library 호환성, graph 구조, 연산자 지원 여부를 초기에 확인해야 합니다.

전체 학습 흐름은 다음과 같습니다.

```text
NPU 추론
→ 서빙
→ 성능 분석
→ 스케줄링 / Cache
→ Tool Calling / Agent
→ 모델 포팅
```
