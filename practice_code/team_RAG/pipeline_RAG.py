import gradio as gr
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import ollama
import pdfplumber
import pytesseract
from PIL import Image
import hashlib
import os
from langchain.document_loaders import PyMuPDFLoader

# 캐시 저장소 (LRU 방식)
retriever_cache = {}
# CACHE_LIMIT = 5

# 파일 해시 생성 함수
def get_file_hash(file):
    file_path = file.name if hasattr(file, "name") else file
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

''' 
PDF에서 텍스트 추출 (PyMuPDFLoader + OCR 결합)
pdfplumber보다 PyMuPDFLoader가 더 빠르고 안정적인 pdf 파싱이 가능
PyMuPDFLoader를 우선 사용후 텍스트가 없는 페이지에만 pytesseract로 OCR 적용하도록 최적화
'''
def extract_text_from_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    if not docs:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or pytesseract.image_to_string(page.to_image())
                docs.append(Document(page_content=text))
    return docs

''' 
텍스트 분할 함수
" ", ". " 등을 넣어서 더 세밀하게 분할
'''
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    return text_splitter.split_documents(docs)

'''
컨텍스트 포매팅 함수
중요 문장 우선 포함 및 길이 제한
'''
def format_context(retrieved_docs, max_length=3000):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return context[:max_length]

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

'''
벡터스토어 생성 및 캐시 관리
메모리 효율성을 높이기 위해서 LRU 방식으로 캐시 관리 기능을 추가할 수 있음.(현재는 주석처리)
'''
def get_vectorstore(docs, file_hash):
    if file_hash in retriever_cache:
        print("📦 캐시된 retriever 사용 중...")
        return retriever_cache[file_hash]

    split_docs = split_text(docs)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    # if len(retriever_cache) >= CACHE_LIMIT:
    #     retriever_cache.pop(next(iter(retriever_cache)))  # 가장 오래된 캐시 제거
    retriever_cache[file_hash] = vectorstore
    return vectorstore

'''
RAG Pipeline 클래스
주어진 쿼리에 대해 PDF 파일에서 컨텍스트를 추출하고, 해당 컨텍스트오 함께 답변을 생성하는 전체 파이프라인
'''
class RAGPipeline:
    def __init__(self, generator):
        self.generator = generator

    def __call__(self, query, file):
        file_hash = get_file_hash(file)
        docs = extract_text_from_pdf(file.name)
        vectorstore = get_vectorstore(docs, file_hash)
        retrieved_docs = vectorstore.similarity_search(query, k=5) # 3에서 5로 바꿈.
        # retrieved_docs = vectorstore.similarity_search_with_score(query, k=5, search_type='mmr')
        formatted_context = format_context(retrieved_docs)
        return self.generator(query, formatted_context)

'''
Generator 클래스
Ollama API를 호출하여 사용자 쿼리와 컨텍스트를 기반으로 자연어 응답 생성
'''
class Generator:
    def __call__(self, query, context):
        formatted_prompt = f"""
        You are a highly accurate document analysis assistant.
        Your task is to provide a precise answer to the user's question based on the provided context.

        Question: {query}

        Context:
        {context}

        Answer with specific and concise information from the context.
        """
        response = ollama.chat(
            model='benedict/linkbricks-llama3.1-korean:8b',
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Analyze the PDF content and answer the question."},
                {"role": "user", "content": formatted_prompt}
            ]
        )
        return response['message']['content']

# Gradio ChatInterface로 UI 설정
generator = Generator()
rag_pipeline = RAGPipeline(generator)

chatbot = gr.ChatInterface(
    fn=lambda msg, hist, file: rag_pipeline(msg, file),
    title="[benedict/linkbricks-llama3.1-korean:8b] Optimized RAG System",
    description="Upload a PDF file and ask questions. The system retrieves relevant context and generates responses.",
    additional_inputs=[gr.File(label="📄 PDF 파일", file_types=[".pdf"])]
)

chatbot.launch()
