# tool calling : 외부 기능을 가져와서 사용하기!!

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip() # 줄바꿈 띄어씌기 인정 안함
base_url = os.environ["MONOROUTER_BASE_URL"].strip()
tavily_api_key = os.environ["TAVILY_API_KEY"].strip()

llm = ChatOpenAI(
    model_name='gpt-6-luna',
    temperature=0,
    max_tokens=1000,
    api_key= api_key,
    base_url=base_url, # 보통 외부키는 이렇게 해야 연결 가능
)


###################################### 0. Tool 정의 ##########################################
# pip install langchain_tavily
from langchain_tavily import TavilySearch

# 웹 검속 도구 생성
web_search =  TavilySearch(
    max_results=3,
)


tools = [web_search]

llm.bind_tools(tools=tools) # llm에 web search 함친거
llm_with_tools = llm.bind_tools(tools=tools)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages # Reducer 
from langgraph.prebuilt import ToolNode, tools_condition

# 메모리 추가 첫번째
from langgraph.checkpoint.memory import InMemorySaver

##################### 1. State 정의 #####################
class State(TypedDict) :
    messages : Annotated[list, add_messages]
    search_result : str
    analysis_result : str
    



#################### 2. node 정의 #####################
# 검색 에이전트
def search_agent(state : State) :
    print("검색 에이전트 실행")
    question = state["messages"][-1].content

    result = web_search.invoke({
        "query" : question
    })

    return {"search_result" : str(result)}

def analysis_agent(state : State) :
    print("분석 에이전트 실행")
    question = state["messages"][-1].content

    response = llm.invoke(
        f"다음 질문을 기존 지식을 바탄으로 분석해줘. \n {question}"
    )

    return {"analysis_result" : response.content}

def merge_agent(state : State) :
    result = (
        state["search_result"] + "\n\n" +
        "==================================" + "\n" +
        state["analysis_result"]
    )
    return {"messages" : [("assistant", result)]}


################### 3. 그래프 구성, 노드 추가 #################
builder = StateGraph(State)
builder.add_node("search", search_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("merge", merge_agent)


################# 병렬 실행 ################
# 구조를 만들어 주면 된다
builder.add_edge(START, "search")
builder.add_edge(START, "analysis")

################# 병합 #####################
builder.add_edge("search", "merge")
builder.add_edge("analysis", "merge")

builder.add_edge("merge", END)


# 컴파일
graph = builder.compile()


# 실행
query = "엔디비아의 미래 전망에 대해 알려줘"

response = graph.invoke({
    "messages" : [("user", query)]
})

print("Assistant : ", response['messages'][-1].content)

###################### 그래프 이미지 ######################
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

# 그림 출력
plt.figure(figsize=(12, 8)) # 통상 인치 / 그림판 사이즈
plt.imshow(img)
plt.axis('off') # 축이 보이지 않음
plt.show()