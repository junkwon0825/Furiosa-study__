from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

response = llm.invoke('현재멜론기준 탑10노래')
# print(response)
print(response.content)







