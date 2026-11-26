from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    #openai_api_key = openai_api_key,
)

response = llm.invoke('현재멜론기준 탑10노래')
# print(response)
print(response.content)







