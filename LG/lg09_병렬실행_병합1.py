from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

#1. state 정의
class State(TypedDict):
    question : str
    search_results : str
    analysis_results : str
    final_results : str


#2. 검색노드
def search_agent(state : State) :
   print("검색 에이전트 실행")
   return {"search_results" : "엔비디아는 AI GPU 시장의 주요 기업이다."} 


#3. 분석노드
def analysis_agent(state : State) :
   print("분석 에이전트 실행")
   return {"analysis_results" : " AI 반도체 수요 증가가 성장 요인이다."}


#4. 결과 통합 노드
def merge_agent(state : State) :
   result = (
      state["search_results"] + "\n " +
      state["analysis_results"]
   )
   return {"final_result" : result} 

#5. 그래프 구성
builder = StateGraph(State)

builder.add_node("search_agent", search_agent)
builder.add_node(analysis_agent)
builder.add_node(merge_agent)

#####################병렬 실행 #################
builder.add_edge(START, "search_agent")
builder.add_edge(START, "analysis_agent")
builder.add_edge(START, "analysis_agent")

#################### 병합 ####################
builder.add_edge("search", "merge")
builder.add_edge("analysis", 'merge')
builder.add_edge("merge", END)

#컴파일
graph = builder.compile()

#실행
response = graph.invoke({
   "question" : "엔비디아의 미래전망은 ?"
})

print(response["final_result"])


import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()  # PNG 파일로 저장

img = Image.open(BytesIO(png_data))

# with open(path + 'lg07_ToolNode_graph.png', 'wb') as file:
#     file.write(png_data)

# file = open(path + 'lg07_ToolNode_graph.png', 'wb')
# file.write(png_data)
# file.close()
# 위에거랑 같은내용

#그림 출력
plt.figure(figsize=(12, 8))    
plt.imshow(img)
plt.axis('off')  # 축 제거
plt.show()
