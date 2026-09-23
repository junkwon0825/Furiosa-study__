# Day 18 — Loss, Learning Rate, Optimizer, Backpropagation, RNN/LSTM

## 1. 딥러닝 학습의 전체 흐름

```text
Forward
→ Prediction
→ Loss
→ 미분
→ Backpropagation
→ Gradient
→ Optimizer
→ Weight Update
→ 다시 Forward
```

핵심은 **Loss가 작아지는 방향으로 Weight와 Bias를 반복해서 수정하는 것**이다.

---

## 2. `model.compile()`에서 시작

```python
model.compile(
    loss='mse',
    optimizer='adam'
)
```

- `loss`: 얼마나 틀렸는지 계산
- `optimizer`: 그 오차를 줄이기 위해 Weight를 어떻게 바꿀지 결정

Adam의 기본 Learning Rate는 보통:

```python
Adam(learning_rate=0.001)
```

---

## 3. Optimization

Loss Function은 **예측과 정답이 얼마나 다른지를 하나의 숫자로 표현하는 함수**이다.

학습 목표:

\[
\boxed{
\min_{\theta} L(\theta)
}
\]

여기서 `θ`는 모델의 모든 Weight와 Bias를 의미한다.

Loss가 작은 Parameter를 찾아가는 과정이 **Optimization**, 이를 수행하는 알고리즘이 **Optimizer**다.

---

## 4. Learning Rate

Learning Rate는 **Weight를 한 번 업데이트할 때 얼마나 크게 움직일지를 정하는 값**이다.

기본 Gradient Descent:

\[
w_{t+1}=w_t-\eta g_t
\]

- `w_t`: 현재 Weight
- `g_t`: 현재 Gradient
- `η`: Learning Rate

```text
Learning Rate가 큼
→ 한 번에 크게 이동
→ 빠를 수 있지만 최적점을 지나칠 수 있음

Learning Rate가 작음
→ 조금씩 이동
→ 안정적이지만 학습이 느릴 수 있음
```

---

## 5. Loss Function 종류

### MSE

회귀에서 자주 사용.

\[
L=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
\]

간단한 손계산에서는:

\[
L=\frac12(\hat y-y)^2
\]

### Binary Cross Entropy

이진분류:

```python
loss='binary_crossentropy'
```

### Categorical Cross Entropy

다중분류 + One-hot Label:

```python
loss='categorical_crossentropy'
```

### Sparse Categorical Cross Entropy

다중분류 + 정수 Label:

```python
loss='sparse_categorical_crossentropy'
```

---

## 6. 딥러닝 학습의 목적

모델:

\[
\hat y=f(x;\theta)
\]

Loss:

\[
L(y,\hat y)
\]

결국:

\[
\boxed{
\theta^*=\arg\min_{\theta}L(y,f(x;\theta))
}
\]

즉 **Loss가 가장 작아지는 Weight와 Bias를 찾는 과정**이다.

---

## 7. 미분과 Gradient

미분:

\[
\frac{\partial L}{\partial w}
\]

의 의미는 **Weight를 조금 바꿨을 때 Loss가 얼마나 변하는지**이다.

변수가 하나일 때는 derivative라고 많이 하고, 변수가 여러 개면 각 변수에 대해 편미분한 값을 모아 Gradient를 만든다.

\[
\nabla L=
\begin{bmatrix}
\frac{\partial L}{\partial w_1}\\
\frac{\partial L}{\partial w_2}\\
\vdots
\end{bmatrix}
\]

Gradient는 Loss가 가장 빠르게 증가하는 방향이므로, Loss를 줄이려면 `-Gradient` 방향으로 이동한다.

---

## 8. Chain Rule

```text
w
↓
ŷ
↓
L
```

Weight가 Loss에 직접 연결되지 않기 때문에 Chain Rule을 사용한다.

\[
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial w}
\]

즉:

```text
Weight가 Prediction에 미치는 영향
×
Prediction이 Loss에 미치는 영향
```

---

## 9. Backpropagation

Backpropagation은 **Chain Rule을 출력에서 입력 방향으로 반복 적용하여 모든 Weight의 Gradient를 계산하는 방법**이다.

```text
Loss
↑
Layer 3 Gradient
↑
Layer 2 Gradient
↑
Layer 1 Gradient
```

정리:

```text
Backpropagation
→ Gradient 계산

Optimizer
→ Gradient를 이용해서 Weight 업데이트
```

---

## 10. Optimizer 종류

### SGD

\[
\boxed{
w_{t+1}=w_t-\eta g_t
}
\]

현재 Gradient만 보고 이동한다.

### Momentum

과거 이동 방향을 기억해 관성을 추가한다.

\[
v_t=\beta v_{t-1}+(1-\beta)g_t
\]

\[
w_{t+1}=w_t-\eta v_t
\]

### RMSProp

Gradient 제곱값의 이동평균을 이용해 Parameter별 Update 크기를 조절한다.

\[
v_t=\beta v_{t-1}+(1-\beta)g_t^2
\]

\[
w_{t+1}
=
w_t-
\eta
\frac{g_t}{\sqrt{v_t}+\epsilon}
\]

### Adam

Adam은 크게 보면 **Momentum + RMSProp**이다.

1차 Moment:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

2차 Moment:

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

초기 0 편향 보정:

\[
\hat m_t=\frac{m_t}{1-\beta_1^t}
\]

\[
\hat v_t=\frac{v_t}{1-\beta_2^t}
\]

최종 업데이트:

\[
\boxed{
w_{t+1}
=
w_t-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
}
\]

대표 기본값:

```text
learning_rate = 0.001
β1 = 0.9
β2 = 0.999
ε = 매우 작은 값
```

Adam은 Learning Rate를 완전히 자동으로 정하는 것이 아니라, **base Learning Rate를 기준으로 실제 Parameter별 Update 크기를 적응적으로 조절**한다.

---

## 11. ReduceLROnPlateau

학습 중 Learning Rate 자체를 줄일 수 있다.

```python
ReduceLROnPlateau(
    factor=0.5,
    patience=5
)
```

예:

```text
0.001
↓
0.0005
↓
0.00025
```

- Adam: Parameter별 실제 Update 크기 적응적 조절
- ReduceLROnPlateau: Base Learning Rate 자체 조절

---

## 12. Loss Landscape

실제 딥러닝에서는:

\[
L(w_1,w_2,\dots,w_n)
\]

형태의 매우 복잡한 고차원 비선형 함수가 된다.

이를 **Loss Landscape**라고 한다.

반드시 Global Minimum만 찾는 것이 목표는 아니다. 실제로는 **충분히 낮은 Loss + 좋은 일반화 성능**을 가진 Parameter를 찾는 것이 중요하다.

Gradient는 1차 미분 정보이고, Hessian은 2차 미분 정보를 모은 행렬이다.

\[
H=\nabla^2L
\]

Hessian은 Loss Surface의 곡률을 나타내지만, Parameter가 많으면 행렬이 너무 커지므로 대규모 딥러닝에서 직접 다루기 어렵다.

---

# RNN

## 13. RNN이란?

RNN = **Recurrent Neural Network**

시간 순서가 중요한 데이터를 처리한다.

예:

```text
온도
주식
채권
센서
음성
문장
시계열
```

---

## 14. RNN 입력 Shape

\[
\boxed{
(batch, timesteps, features)
}
\]

예:

```text
(1000, 10, 5)
```

- 1000: Sample 수
- 10: Timestep
- 5: 각 Timestep의 Feature 수

---

## 15. 시계열에서 x, y 만들기

Raw Data:

```text
1, 2, 3, 4, 5, 6, 7 ...
```

Window 방식:

```text
x = [1,2,3] → y = 4
x = [2,3,4] → y = 5
x = [3,4,5] → y = 6
```

즉 과거 일정 구간을 입력으로 사용해 다음 값을 정답으로 만든다.

---

## 16. SimpleRNN 핵심식

\[
\boxed{
h_t
=
\tanh(
x_tW_x+
h_{t-1}W_h+
b
)
}
\]

- `x_t`: 현재 입력
- `h_(t-1)`: 이전 Hidden State
- `W_x`: Input → Hidden Weight
- `W_h`: Hidden → Hidden Recurrent Weight
- `b`: Bias

---

## 17. RNN Weight Sharing

RNN에서는 Timestep마다 새로운 Weight를 만드는 것이 아니다.

```text
t1 → 같은 W_x, W_h
t2 → 같은 W_x, W_h
t3 → 같은 W_x, W_h
...
```

같은 Weight Matrix를 모든 Timestep에서 공유한다.

---

## 18. SimpleRNN Parameter 수

입력 Feature 수를 `F`, Hidden Unit 수를 `U`라고 하면:

\[
FU+U^2+U
\]

즉:

\[
\boxed{
U(F+U+1)
}
\]

예:

```text
features = 3
units = 2
```

\[
2(3+2+1)=12
\]

---

## 19. RNN의 Weight와 Bias

예를 들어 Feature=3, Units=2라면:

\[
W_x\in\mathbb{R}^{3\times2}
\]

\[
W_h\in\mathbb{R}^{2\times2}
\]

\[
b\in\mathbb{R}^{2}
\]

Weight는 Matrix, Bias는 보통 Vector다.

Bias는 **뉴런 출력의 기준점을 이동시키는 학습 Parameter**다.

---

## 20. BPTT

RNN의 역전파는 시간축을 따라 뒤로 진행된다.

이를 **Backpropagation Through Time**이라고 한다.

```text
Forward
x1 → h1 → h2 → h3 → h4

Backward
Loss
↓
h4
↓
h3
↓
h2
↓
h1
```

---

## 21. Vanishing Gradient

긴 Sequence에서 작은 미분값이 계속 곱해지면:

\[
0.5^{100}
\]

처럼 Gradient가 거의 0이 될 수 있다.

```text
작은 값 계속 곱함
↓
Gradient 거의 0
↓
초기 Timestep의 정보 학습 어려움
```

이것이 **Vanishing Gradient**다.

반대로 큰 값이 계속 곱해져 Gradient가 지나치게 커지는 것은 **Exploding Gradient**다.

---

# LSTM

## 22. LSTM이란?

LSTM = **Long Short-Term Memory**

기본 RNN의 장기 의존성 문제와 Vanishing Gradient 문제를 완화하기 위해 만들어졌다.

```text
SimpleRNN
→ Hidden State

LSTM
→ Hidden State
+ Cell State
```

Cell State는 장기 기억 통로 역할을 한다.

---

## 23. LSTM Gate

LSTM은 대표적으로:

```text
Forget Gate
→ 무엇을 잊을지

Input Gate
→ 무엇을 새로 기억할지

Output Gate
→ 무엇을 출력할지
```

를 사용한다.

Cell State 업데이트:

\[
C_t
=
f_t\odot C_{t-1}
+
i_t\odot\tilde C_t
\]

의미:

```text
새 Cell State
=
남겨둘 이전 기억
+
새로 저장할 기억
```

---

## 24. SimpleRNN vs LSTM

```text
SimpleRNN
→ 구조 단순
→ Parameter 적음
→ 긴 Sequence에서 Gradient 문제 큼

LSTM
→ 구조 복잡
→ Parameter 많음
→ 장기 의존성 학습에 더 유리
```

단, LSTM이 모든 문제에서 무조건 최고인 것은 아니다. 문제에 따라 SimpleRNN, GRU, LSTM, Transformer 등을 선택한다.

---

# 25. 오늘의 최종 연결

```text
Model
↓
Forward
↓
Prediction
↓
Loss
↓
미분
↓
Backpropagation
↓
Gradient
↓
Optimizer
↓
Learning Rate 적용
↓
Weight / Bias Update
↓
다시 Forward
```

RNN도 동일한 학습 원리를 사용한다.

차이는:

```text
RNN
→ 시간축에서 같은 Weight 반복 사용

BPTT
→ 시간축을 따라 Gradient 역전파
```

---

# 26. 오늘의 핵심 공식

### SGD

\[
\boxed{
w_{t+1}=w_t-\eta g_t
}
\]

### Adam

\[
\boxed{
w_{t+1}
=
w_t-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
}
\]

### SimpleRNN

\[
\boxed{
h_t
=
\tanh(
x_tW_x+
h_{t-1}W_h+
b
)
}
\]

### SimpleRNN Parameter Count

\[
\boxed{
U(F+U+1)
}
\]

---

# 27. 오늘의 한 줄

> **딥러닝 학습은 Loss를 줄이기 위해 Backpropagation으로 Gradient를 계산하고 Optimizer가 Weight와 Bias를 갱신하는 반복 과정이며, RNN은 이 동일한 원리를 시간축 방향으로 확장한 모델이다.**
