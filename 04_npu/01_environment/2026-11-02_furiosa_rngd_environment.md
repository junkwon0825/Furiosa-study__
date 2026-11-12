# 2026-11-02 Furiosa NPU Study

## 1. Furiosa RNGD NPU 환경 확인

오늘부터 Furiosa RNGD NPU 실습을 시작했다.

기존 GPU 기반 학습에서 한 단계 더 나아가, AI 모델을 NPU 환경에서 추론하고 이후에는 모델 최적화와 serving, latency/throughput benchmark까지 진행할 예정이다.

NPU 서버 접속 후 장치 상태는 다음 명령으로 확인할 수 있다.

```bash
furiosa-smi
furiosa-smi status
furiosa-smi info
furiosa-smi ps
```

실시간 상태 확인:

```bash
watch -n 1 furiosa-smi status
```

---

## 2. Hugging Face 공유 Token 문제

RNGD 서버의 container 환경에서는 `~/.cache`가 container 전용 disk가 아니라 host disk에 mount되어 있을 수 있다.

이 경우 Hugging Face CLI의 기본 token 저장 위치인

```text
~/.cache/huggingface/token
```

도 다른 container와 공유될 가능성이 있다.

즉, 내가 로그인하지 않았는데 다른 사용자의 Hugging Face 계정이 나타나는 문제가 발생할 수 있다.

### Disk 구조 확인

```bash
findmnt -no SOURCE -T ~
findmnt -no SOURCE -T ~/.cache
findmnt -no SOURCE -T ~/.local
df -h ~
```

확인 결과:

```text
HOME       → /dev/rbd0
~/.cache   → /dev/nvme1n1[/cache]
```

`HOME`과 `~/.cache`가 서로 다른 disk에 위치하고 있었다.

따라서 `~/.cache`는 host/shared storage일 가능성이 있고, 인증정보를 이 위치에 저장하지 않도록 분리할 필요가 있다.

---

## 3. Hugging Face 인증정보를 Container Disk로 분리

개인 영역에 사용할 directory를 생성했다.

```bash
install -d -m 700 ~/.local/cache ~/.local/share/huggingface
```

권한 확인:

```bash
ls -ld ~/.local ~/.local/cache ~/.local/share ~/.local/share/huggingface
```

Hugging Face token이 저장될 directory는 본인만 접근할 수 있도록 `700` 권한으로 설정했다.

```text
/root/.local/cache
/root/.local/share/huggingface
```

---

## 4. Cache와 Token 분리 전략

목표 구조:

```text
개인 인증 정보
/root/.local/share/huggingface/token
        │
        └── container 전용 disk

모델 cache
/root/.cache/huggingface/hub
        │
        └── shared host disk
```

큰 모델 파일은 여러 container에서 공유할 수 있도록 유지하면서, 계정 token만 개인 disk에 저장한다.

`.bashrc`에는 다음 환경변수를 설정한다.

```bash
export XDG_CACHE_HOME=$HOME/.local/cache
export HF_HOME=$HOME/.local/share/huggingface
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH
```

각 변수의 목적:

- `XDG_CACHE_HOME`: 일반 프로그램 cache를 개인 disk로 이동
- `HF_HOME`: Hugging Face token 및 개인 설정 저장
- `HF_HUB_CACHE`: 모델 weight는 기존 shared cache 사용
- `unset ...`: 기존 환경변수에 남아 있는 다른 사용자의 token 사용 방지

---

## 5. Hugging Face 로그인 확인

환경 설정 후 새로운 shell에서 token 저장 위치를 확인한다.

```bash
python3 -c "from huggingface_hub import constants as c; print(c.HF_TOKEN_PATH)"
```

예상 경로:

```text
/root/.local/share/huggingface/token
```

그 후 개인 Hugging Face 계정으로 로그인한다.

```bash
hf auth login
hf auth whoami
```

Token 자체는 GitHub나 코드에 절대 저장하지 않는다.

---

## 6. 오늘 배운 핵심

### GPU와 NPU

GPU는 범용 병렬 연산에 강하고 AI 학습과 추론 모두에 널리 사용된다.

NPU는 Neural Network 연산에 특화된 accelerator이며, 앞으로 RNGD에서 다음 내용을 학습할 예정이다.

```text
Model
  ↓
Compile / Optimization
  ↓
NPU Runtime
  ↓
Inference
  ↓
Serving
  ↓
Benchmark
```

주요 성능 지표:

- Latency
- Throughput
- TTFT
- TPS
- Batch size
- NPU utilization

---

## 7. K-Fold Cross Validation

K-Fold는 전체 데이터를 K개로 나눈 뒤 한 fold를 validation으로 사용하고 나머지를 training에 사용하는 과정을 K번 반복하는 방법이다.

예를 들어 `K=5`라면:

```text
1회: Fold1 Validation / Fold2~5 Train
2회: Fold2 Validation / Fold1,3~5 Train
3회: Fold3 Validation / 나머지 Train
4회: Fold4 Validation / 나머지 Train
5회: Fold5 Validation / 나머지 Train
```

각 fold의 validation score를 평균내 모델의 일반화 성능을 좀 더 안정적으로 평가할 수 있다.

분류 문제에서는 클래스 비율을 유지하는 `StratifiedKFold`도 많이 사용한다.

---

## Next

- Hugging Face 개인 token 환경 설정 완료
- RNGD에서 Hugging Face model 다운로드
- NPU inference 실행
- 모델 compile 과정 확인
- GPU와 RNGD 성능 비교
- Latency / Throughput benchmark