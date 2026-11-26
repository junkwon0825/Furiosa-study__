# 05. AI Agent

## 학습 내용

LangGraph를 사용해 LLM 호출을 단순 text generation에서 상태 기반 Agent workflow로 확장했습니다.

```text
사용자 입력
→ Graph State
→ LLM Node
→ 조건 분기
→ ToolNode
→ Tool Result
→ LLM
→ 최종 응답
```

## 핵심 개념

- `TypedDict`, `Annotated`, `add_messages` 기반 state 정의
- `StateGraph`, node, edge, conditional edge
- `ToolNode`, `tools_condition`
- DuckDuckGo/Tavily 검색 도구 연결
- `InMemorySaver`를 사용한 대화 memory
- 여러 branch의 병렬 실행과 결과 병합

## 파일 흐름

| 범위 | 내용 |
|---|---|
| `langgraph/lg01~02` | State schema와 message reducer |
| `langgraph/lg03~04` | Chatbot graph와 node/edge 구성 |
| `langgraph/lg05~07` | ToolNode, 검색 도구, 조건부 routing |
| `langgraph/lg08_*` | Checkpoint memory |
| `langgraph/lg09_*` | 병렬 branch와 결과 병합 |

RNGD에서 서빙한 LLM에 산술 도구와 multi-step Agent를 연결한 실습은 [`06_npu/11_tool_calling`](../06_npu/11_tool_calling/)에 별도로 보존했습니다.
