from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = 'ss'
#환경변수에 키 넣으니까 그냥 돌아가는거 모델에 넣은게 아님/

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    #openai_api_key = openai_api_key,
)

response = llm.invoke('현재멜론기준 탑10노래')
# print(response)
print(response.content)







