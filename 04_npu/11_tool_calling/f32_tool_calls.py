from kingkeras import get_llm
from langchain_core.tools import tool

#1. 도구 정의
@tool
def add(aaa : int, bbb : int) -> int:
    """두 정수를 더합니다."""
    return aaa + bbb

print(add.name) #함수명
print(add.description) # add함수의 주석
print(add.args)

#2. llm 생성
llm = get_llm(max_tokens=200)

#3. 도구 등록 
llm_with_tools = llm.bind_tools([add])

#4. 질문
response = llm_with_tools.invoke(
    "237341과 3123321을 add로 더해줘."
)

#5. 결과 확인
for call in response.tool_calls:
    print("tool 이름 : ", call['name'])
    print("매개 변수 : ", call['args']) 
    print("호출 ID : ", call['id']) 

# tool 이름 :  add
# 매개 변수 :  {'aaa': 237341, 'bbb': 3123321}
# 호출 ID :  call-fd527cc1d91b46feb7a72a9a34eb8417

