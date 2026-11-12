# Troubleshooting

NPU 실습에서 발생한 오류를 단순 로그가 아니라 **문제 → 원인 → 확인 → 해결 → 배운 점** 순서로 기록합니다.

## torch.compile backend

### InvalidBackend
잘못된 backend 이름을 직접 문자열로 전달하면서 발생.

### ImportError
Furiosa Torch 버전에 따라 import API가 달라질 수 있으므로 설치된 SDK API를 기준으로 확인.

## ResNet18

### undefined symbol
PyTorch / torchvision / Furiosa extension의 binary compatibility 확인 필요.

### unsupported ATen operator
`aten::_native_batch_norm_legit_no_training` 등 NPU에서 직접 지원되지 않는 연산 확인.

대응 후보:
- BatchNorm freeze / fold
- graph decomposition
- 지원 연산으로 치환

## GGUF Quantization

입력 GGUF 파일 위치와 현재 작업 디렉터리가 달라 `gguf_init_from_file` 실패.

상위 디렉터리 파일을 사용할 경우 상대경로를 정확히 지정.

## Furiosa LLM Serving

### Connection refused
LangChain 코드 문제가 아니라 `127.0.0.1:8000`의 OpenAI-compatible server가 실행되지 않은 상태.

### Device mismatch
실제 장치가 `npu0`인데 다른 device id를 지정하지 않도록 `furiosa-smi info`로 먼저 확인.

## Rule

오류를 해결하면 아래 다섯 항목을 추가합니다.

1. Symptom
2. Root cause
3. Diagnostic command
4. Fix
5. Lesson learned
