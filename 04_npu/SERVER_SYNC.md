# NPU Server Sync Guide

NPU 서버에서 진행한 실습 코드를 GitHub의 `04_npu/` 구조로 동기화하는 절차입니다.

## 1. Repository 확인

```bash
cd ~
git clone https://github.com/junkwon0825/Furiosa-study__.git
cd Furiosa-study__
git pull origin main
```

이미 clone 되어 있다면:

```bash
cd ~/Furiosa-study__
git status
git pull --rebase origin main
```

## 2. NPU 수업 코드 복사

현재 서버 수업 경로 예시:

```text
/root/furiosa_study/furiosa2/
```

LLM serving 코드:

```bash
mkdir -p ~/Furiosa-study__/04_npu/05_llm_serving
cp /root/furiosa_study/furiosa2/f21_continuous_batching1.py ~/Furiosa-study__/04_npu/05_llm_serving/
cp /root/furiosa_study/furiosa2/kingkeras.py ~/Furiosa-study__/04_npu/05_llm_serving/
```

전체 파일을 무작정 복사하기 전에 아래를 먼저 확인합니다.

```bash
find /root/furiosa_study/furiosa2 -maxdepth 2 -type f | sort
```

## 3. Secret 검사

```bash
grep -RniE "api[_-]?key|token|password|secret|hf_[A-Za-z0-9]+" 04_npu || true
```

실제 token이나 API key가 들어간 파일은 push하지 않습니다.

## 4. Commit & Push

```bash
git status
git add 04_npu
git commit -m "add NPU server study codes"
git push origin main
```

## Recommended Mapping

```text
NPU 환경/장치 확인        → 04_npu/01_environment/
torch.compile 실습        → 04_npu/02_torch_compile/
ResNet/YOLO 포팅          → 04_npu/03_model_porting/
GGUF/양자화               → 04_npu/04_quantization/
furiosa-llm/LangChain     → 04_npu/05_llm_serving/
latency/throughput 측정   → 04_npu/06_benchmark/
```
