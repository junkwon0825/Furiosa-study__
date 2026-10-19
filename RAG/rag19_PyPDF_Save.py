# transformer_paper_save.py
# Attention Is All You Need PDF -> Chunking -> Embedding -> Chroma 저장

import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

# 01. PDF 불러오기
path = "./_data/"
pdf_loader = PyPDFLoader(path + "Attention is all you need.pdf")
pdf_docs = pdf_loader.load()

print("PDF 페이지 수 :", len(pdf_docs))
print(pdf_docs[0].page_content[:500])

# PyPDFLoader의 page는 0부터 시작하므로 사람이 보는 페이지 번호로 +1
for doc in pdf_docs:
    doc.metadata["page_num"] = doc.metadata.get("page", 0) + 1

# 02. Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)

texts = text_splitter.split_documents(pdf_docs)

print("생성된 Chunk 수 :", len(texts))
print("첫 번째 Chunk 페이지 :", texts[0].metadata["page_num"])

# 03. Embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
)

# 04. Chroma 저장
DB_PATH = "./_db/Transformer_Paper/"
COLLECTION_NAME = "attention_is_all_you_need"

vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name=COLLECTION_NAME,
)

print("Chroma 저장 완료")
print("벡터 저장소 문서 수 :", vector_store._collection.count())
print("저장 위치 :", DB_PATH)
