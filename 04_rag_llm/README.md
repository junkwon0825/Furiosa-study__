# 04. LLM / RAG

## 학습 내용

LangChain을 사용해 prompt와 LLM 호출부터 embedding, vector store, retriever, RAG chain, PDF 질의응답, hybrid search까지 구현했습니다.

```text
Document
→ Chunking
→ Embedding
→ Vector Store / Search
→ Retriever
→ LLM
→ Answer
```

## 핵심 개념과 실험

- LCEL: prompt → model → output parser
- OpenAI/Hugging Face embedding
- Chroma 저장·불러오기와 similarity search
- FAISS index 저장·불러오기
- `TextLoader`, `PyPDFLoader`, `RecursiveCharacterTextSplitter`
- Retrieval chain과 Gradio chatbot
- BM25, vector search, `EnsembleRetriever` 기반 hybrid retrieval

## 대표 파일

| 범위 | 내용 |
|---|---|
| `code/rag07~09` | Prompt, LCEL, output parser |
| `code/rag10~14` | Embedding, Chroma, RAG pipeline |
| `code/rag17~18` | FAISS 저장·검색과 Gradio 연동 |
| `code/rag19~20` | PDF와 embedding model 기반 RAG |
| `code/rag22_*` | 문서, 유사도, prompt 확인 |
| `code/rag23_*` | BM25 / vector / hybrid search |

API key 값, `.env`, vector DB 파일, embedding cache는 저장소에 포함하지 않았습니다.
