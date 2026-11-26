# 11. Tool Calling과 Agent

이 폴더는 하나의 LangChain 도구를 정의하는 기초 단계부터 RNGD에서 서빙하는 LLM을 연결한 다단계 CLI Agent까지의 구현 과정을 순서대로 보존합니다.

| 파일 | 학습 내용 |
|---|---|
| `f31_tool_test.py` | 도구를 등록하고 모델 응답 확인 |
| `f32_tool_calls.py` | 도구 이름, 인자, call ID 확인 |
| `f33_tool_invoke.py` | 모델이 선택한 Python 도구 실행 |
| `f34_multi_tools.py` | 여러 산술 도구 등록 및 분기 실행 |
| `f35_agent01.py`, `f35_agent02.py` | `create_agent` 도입과 Agent message 확인 |
| `f36_agent_tool.py` | 네 가지 산술 도구를 Agent에 연결 |
| `f37_config.py` | `recursion_limit`을 사용한 다단계 실행 제어 |
| `f38_agent_error.py` | 도구 및 Agent 실행 오류 처리 |
| `f39_agent_project.py` | Tool call 추적 기능을 포함한 대화형 CLI Agent |
| `kingkeras.py` | RNGD LLM 연결 helper와 tool-choice 서빙 설정 |

구현한 도구는 `add`, `subtract`, `multiply`, `divide`입니다.

```text
사용자 → RNGD 서빙 LLM → Agent → 도구 선택 → Python 실행
      ← 최종 응답       ← 도구 결과 ←──────────────────
```

각 파일은 수업에서 직접 진행한 실습 흐름을 보존하며 API key나 모델 weight는 포함하지 않습니다.
