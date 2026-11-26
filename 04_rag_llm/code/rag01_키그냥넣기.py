from langchain_openai import ChatOpenAI

openai_api_key = 'dd'
#openai_api_key는 꼭 숨겨야된다.

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    openai_api_key = openai_api_key,
)

response = llm.invoke('최근 자율주행자동차 발전과 미래 향후 발전기획은어때')
# print(response)
print(response.content)







