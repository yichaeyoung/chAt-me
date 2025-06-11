# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from langchain_community.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# import os

# ''' PDF 경로 입력 '''
# pdf_dir = "../dataset"
# pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

# ''' 문서 호출, 분할 '''
# all_docs = []
# for path in pdf_paths:
#     loader = PyMuPDFLoader(path)
#     docs = loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     split_docs = splitter.split_documents(docs)
#     all_docs.extend(split_docs)

# ''' 임베딩 모델 준비 '''
# embedding_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# ''' Chroma_db에 저장 '''
# db_path = "../dataset"
# vectorstore = Chroma.from_documents(documents=all_docs, embedding=embedding_model, persist_directory=db_path)

# ''' 저장 완료 '''
# vectorstore.persist()
# print(f"저장 완료! {len(all_docs)}개 문서가 Chroma DB에 저장됨")
import os
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_DB_DIR = "../dataset"
CSV_SAVE_DIR = "../dataset"
os.makedirs(CSV_SAVE_DIR, exist_ok=True)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def save_documents_to_csv(documents, file_hash):
    csv_path = os.path.join(CSV_SAVE_DIR, f"{file_hash}.csv")
    rows = []
    for doc in documents:
        rows.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return csv_path

def build_chroma_db(documents, file_hash):
    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    return vectorstore
