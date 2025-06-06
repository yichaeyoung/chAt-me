from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

''' PDF 경로 입력 '''
pdf_dir = "docs"
pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

''' 문서 호출, 분할 '''
all_docs = []
for path in pdf_paths:
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    all_docs.extend(split_docs)

''' 임베딩 모델 준비 '''
embedding_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

''' Chroma_db에 저장 '''
db_path = "./chroma_db"
vectorstore = Chroma.from_documents(documents=all_docs, embedding=embedding_model, persist_directory=db_path)

''' 저장 완료 '''
vectorstore.persist()
print(f"✅ 저장 완료! {len(all_docs)}개 문서가 Chroma DB에 저장됨")
