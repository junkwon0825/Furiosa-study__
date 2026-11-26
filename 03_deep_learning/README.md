# 03. Deep Learning

## 학습 내용

TensorFlow/Keras를 사용해 기본 Dense network에서 CNN, RNN/LSTM, Conv1D, transfer learning과 학습 모니터링까지 단계적으로 실습했습니다.

```text
데이터 전처리
→ 모델 구조 정의
→ compile / train
→ validation
→ evaluation
→ callback / TensorBoard monitoring
```

## 핵심 개념

- Sequential / Functional API
- Regression, binary classification, multiclass classification
- Scaling과 data leakage 방지
- EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Dropout과 model/weight save-load
- Conv2D, pooling, augmentation, CNN image classification
- SimpleRNN, LSTM, Bidirectional, Conv1D
- Tokenizer와 Embedding
- VGG16 기반 transfer-learning 실습
- TensorBoard logging

## 폴더

| 경로 | 내용 |
|---|---|
| `keras_basics/` | Dense부터 CNN/RNN까지의 초기 순차 실습 |
| `keras_advanced/` | Embedding, Conv1D, multi-input/output, transfer learning, TensorBoard |
| `dataset_index.txt` | 파일 번호에 사용한 대표 dataset 매핑 |

수업 코드가 많은 영역이므로 파일 번호를 유지해 학습 순서를 보존했습니다. 생성된 model history CSV와 날짜별 강의 노트는 Git에서 제거하고, 기술 흐름은 이 README와 [`docs/learning_journey.md`](../docs/learning_journey.md)에 통합했습니다.
