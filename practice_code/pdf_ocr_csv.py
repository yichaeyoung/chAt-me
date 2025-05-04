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

# OCR로 텍스트 추출
def extract_text_with_ocr(page):
    text = page.extract_text()
    if not text:
        image = page.to_image()
        text = pytesseract.image_to_string(image)
    return text

# PDF에서 표 추출 후 CSV 저장
def extract_tables_from_pdf(file):
    markdown_tables = []
    merged_tables = []
    try:
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                    if not tables:
                        continue
                    for table in tables:
                        df = pd.DataFrame(table)
                        df.columns = df.iloc[0]
                        df = df[1:]
                        merged_tables.append(df)
                        markdown_tables.append(df.to_markdown(index=False))
                except Exception as e:
                    print(f"테이블 추출 에러 (페이지 {i + 1}): {e}")
    except Exception as e:
        print(f"PDF 열기 실패: {e}")
        return ""

    if merged_tables:
        result = pd.concat(merged_tables, ignore_index=True)
        result.to_csv("tables.csv", index=False)
        print(f"tables.csv 저장 완료 (총 {len(merged_tables)}개 테이블)")
        return "\n\n".join(markdown_tables)
    else:
        print("테이블이 추출되지 않았습니다.")
        return ""

# PDF 로드 및 벡터화 + 테이블 추출
def load_and_retrieve_docs(file):
    file_hash = get_file_hash(file)

    if file_hash in retriever_cache:
        print("캐시된 retriever 사용 중...")
        return retriever_cache[file_hash]

    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = extract_text_with_ocr(page)
                if page_text:
                    text += page_text
    except Exception as e:
        return f"PDF 읽기 오류: {e}"

    if not text:
        return "PDF에서 텍스트를 추출하지 못했습니다."

    # 벡터화
    docs = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    retriever_cache[file_hash] = retriever
    return retriever

# 문서 리스트를 텍스트로 포맷팅
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 기반 QA
def rag_chain(message, history, file):
    # 테이블 추출 및 저장
    table_markdown = extract_tables_from_pdf(file)

    # 텍스트 기반 검색용 벡터화
    retriever = load_and_retrieve_docs(file)
    if isinstance(retriever, str):  # 에러 문자열 반환된 경우
        return retriever

    # 관련 문서 검색
    retrieved_docs = retriever.invoke(message)
    formatted_context = format_docs(retrieved_docs)

    # LLM 질문 포맷
    formatted_prompt = f"""
Question: {message}

Context: {formatted_context}

Tables:\n{table_markdown}
"""

    response = ollama.chat(
        model='llama3.2',
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the given PDF content and tables to answer the question."},
            {"role": "user", "content": formatted_prompt}
        ]
    )

    summary = response['message']['content']
    save_to_csv(summary)
    return summary

# 요약 결과 저장
def save_to_csv(summary):
    df = pd.DataFrame({"Summary": [summary]})
    df.to_csv("summary.csv", index=False)

# Gradio 인터페이스 구성
chatbot = gr.ChatInterface(
    fn=rag_chain,
    title="[LLAMA 3.2] RAG + Table 추출 챗봇 시스템",
    description="PDF파일을 업로드하고 질문을 입력하면 답변을 생성합니다. (파일은 캐시에 저장되고, 테이블은 tables.csv로 저장됩니다.)",
    additional_inputs=[gr.File(label="📄 PDF 파일", file_types=[".pdf"])]
)

chatbot.launch()
