import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"
"""
from glob import glob
path = './_data/rag_data/'

#폴더에서 텍스트 파일 목록 가져오기
txt_files = glob(os.path.join(path, '*.txt'))
print(txt_files)

#데이터 불러온다
data = []

for text_file in txt_files:
    loader = TextLoader(text_file, encoding='utf-8')
    data += loader.load()
print(data[0].page_content)

char_count = [len(doc.page_content) for doc in data]
print(char_count) #[8158, 2049, 1898]

#문서를 자른다
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    separators=["\n\n", "\n", " ", ""], #통상 디폴트값
)

texts = text_splitter.split_documents(data)
# print("생성된 텍스트 청크수 :", len(texts))
# print("각 청크의 길이 :", list(len(text.page_content)for text in texts))
# print("첫번쨰 청크의 내용 :", texts[0].page_content) #한국의 AI for All 프로젝트' metadata={'source': './_data/rag_data\\2026_AI_for_All.txt'}
# print("두번쨰 청크의 내용 :", texts[1].page_content)
"""
#03. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url=base_url,
        # dimensions=5,
)## dimension조절 가능
# sample_text = "삼성전자의 창업자는 누구인가요?"
# vector = embeddings.embed_query(sample_text)
# print(len(vector))


DB_PATH = './_db/Chroma12/' 
# #저장
# vector_store = Chroma.from_documents(
#     documents=texts,
#     embedding=embeddings,
#     persist_directory=DB_PATH,
#     collection_name='chroma12',
# )
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma12',
)
print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}")

query = " 삼성전자의 창업자는 누구인가요 "
result = vector_store.similarity_search(query)

print(f"검색 결과의 길이: {len(result)}")

################## Retrievers #################
################# 검색기 ################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever)
aaa = retriever.invoke(query)
print(f"검색된 관련 문서수 : {len(aaa)}")
print(f"첫번쨰 관련 문서내용 미리 보기 : {aaa[0].page_content[:50]}")






