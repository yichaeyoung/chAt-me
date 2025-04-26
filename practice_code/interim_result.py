import gradio as gr
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama
import pdfplumber
import pytesseract
from PIL import Image

import hashlib
import os

# 캐시 저장소
retriever_cache = {}

# 파일 해시 생성 함수
def get_file_hash(file):
    file_path = file.name if hasattr(file, "name") else file
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# PDF page에서 텍스트 추출하는 함수
def extract_text_with_ocr(page):
    text = page.extract_text()
    if not text:  # 만약 추출할 텍스트가 없다면
        image = page.to_image()
        text = pytesseract.image_to_string(image)
    return text

# PDF 파일을 열어서 extract_text_with_ocr 함수 실행 -> 벡터 데이터베이스에 저장하는 함수
def load_and_retrieve_docs(file):
    file_hash = get_file_hash(file)

    # 캐시 확인
    if file_hash in retriever_cache:
        print("📦 캐시된 retriever 사용 중...")
        return retriever_cache[file_hash]

    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = extract_text_with_ocr(page)
                if page_text:
                    text += page_text
    except Exception as e:
        return f"Error reading PDF file: {e}"

    if not text:
        return "No text found in the PDF file."

    docs = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 캐시에 저장
    retriever_cache[file_hash] = retriever
    return retriever

# 리스트 안의 모든 document 객체 내용을 추출해서 string으로 이어붙여 반환
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
def rag_chain(message, history, file):
    retriever = load_and_retrieve_docs(file)
    if isinstance(retriever, str):
        return retriever

    retrieved_docs = retriever.get_relevant_documents(message)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {message}\n\nContext: {formatted_context}"
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Check the pdf content and answer the question."},
            {"role": "user", "content": formatted_prompt}
        ]
    )

    summary = response['message']['content']
    save_to_csv(summary)
    return summary

# 요약 텍스트를 CSV 파일에 저장하는 함수
def save_to_csv(summary):
    df = pd.DataFrame({"Summary": [summary]})
    df.to_csv("summary.csv", index=False)

# Gradio ChatInterface로 UI 변경
chatbot = gr.ChatInterface(
    fn=rag_chain,
    title="[LLAMA 3.2] RAG 검색 활용 챗봇 시스템",
    description="PDF파일을 업로드하고 질문을 입력하면 답변을 생성해드립니다. (파일은 캐시에 저장됩니다.)",
    additional_inputs=[gr.File(label="📄 PDF 파일", file_types=[".pdf"])]
)

chatbot.launch()
