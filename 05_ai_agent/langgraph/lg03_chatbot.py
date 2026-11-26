from urllib import response

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    max_tokens=2000,
    api_key=api_key,
    base_url=base_url,
)

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

################ 1. state 정의 ###################
class State(TypedDict):
    messages : Annotated[list, add_messages]  # messages 필드에 add_messages 데코레이터 적용
    aaa : str

# 위에꺼는 한마디로 아래다!
state = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    "aaa": "This is a sample string."
}

############### 2. node 정의 ###################
def chatbot(state : State) :
    return {"messages" : [llm.invoke(state["messages"])]}

############3. graph 정의 ###################
graph_01 = StateGraph(State)

graph_01.add_node("chatbot", chatbot)

############ 4. edge 정의 ###################
graph_01.add_edge(START, "chatbot")

graph_01.add_edge("chatbot", END)

############ 5. graph compile ###################
graph = graph_01.compile()

############# 6. graph 실행 ###################
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
    "messages": [("user", query)],
})

# print(response)
# print(response["messages"][0])    
print(response["messages"][1])    
