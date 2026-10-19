import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma


from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

# #데이터 불러온다
# path = './_data/rag_data/'
# loader1 = TextLoader(path + "samsung_outlook.txt", encoding='utf-8')
# loader2 = TextLoader(path + "nvidia_outlook.txt", encoding='utf-8')

# #문서를 자른다
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 300,
#     chunk_overlap = 100,
#     separators=["\n\n", "\n", " ", ""],
# )

# # 문서분할 자른다 / chunking

# split_doc1 = loader1.load_and_split(text_splitter) #chunk 300, overlap 100
# split_doc2 = loader2.load_and_split(text_splitter) #chunk 300, overlap 100

# print(split_doc1)
# print(len(split_doc1), len(split_doc2)) #9 9

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url=base_url,
        # dimensions=5,
)## dimension조절 가능

DB_PATH = './_db/Chroma/' 

db = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma11',
)

#저장된 데이터 확인
print("=================================")
print(db.get())

print("=================================")
aaa = db.similarity_search("삼성전자 사업전망에 대해 알려줘", k=2) #디폴트 4
print(aaa)






