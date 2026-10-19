#10-1 copy
# LCE = Langchain Expressipn Language
#chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = "삼성전자의 창업주는 누구인가요?"
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-small",
#         api_key=api_key,
#         base_url=base_url,
# )

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3",
    model_kwargs={
        "device" : "cpu",
        # "local_files_only" : True
    }
)

vector = embeddings.embed_query(prompt)
print(vector)
print("=================================")
print("임베딩 벡터의 차원: ", len(vector)) #1024

# chain = prompt | model

# input = {"topic" : "양자컴퓨터 학습 원리", "how" : "석사수준"}

# response = chain.invoke(input)
# print(response.content)

