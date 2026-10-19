#14-2copy

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"


#03. 임베딩
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
)

DB_PATH = './_db/Faiss17'


# ==================== FAISS 불러오기 ====================

vector_store = FAISS.load_local(
    DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print(
    f"벡터 저장소에 저장된 문서 수 : "
    f"{vector_store.index.ntotal}"
)


################## Retrievers #################
################# 검색기 ################

retriever = vector_store.as_retriever(
    search_kwargs={"k":2}
)

print(retriever)


print("==========================================================")


################################## 모델 연결 #####################################

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5-nano-chat",
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)


print("==========================================================")


############################## Prompt / RAG Chain ##############################

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요.
컨텍스트에 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다."라고 말씀해 주세요.

컨텍스트 : {context}

질문 : {input}

답변:
""")


# 문서 + Prompt + Model
docu_chain = create_stuff_documents_chain(
    model,
    prompt
)


# Retriever + document chain
rag_chain = create_retrieval_chain(
    retriever,
    docu_chain
)


################################ gradio 챗봇 #########################

import gradio as gr

def answer_invoke(message, history):

    response = rag_chain.invoke(
        {"input": message}
    )

    return response['answer']


demo = gr.ChatInterface(
    fn=answer_invoke,
    title='AI chat-bot'
)

demo.launch()