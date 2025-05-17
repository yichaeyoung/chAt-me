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
import camelot
from langchain.document_loaders import PyMuPDFLoader
import re
import io
import numpy as np

# 캐시 저장소 (LRU 방식)
retriever_cache = {}
# CACHE_LIMIT = 5

# 파일 해시 생성 함수
def get_file_hash(file):
    ''' 파일 해시 생성 함수 - 파일이 존재하지 않으면 None 반환 '''
    try:
        file_path = file.name if hasattr(file, "name") else file
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        print("[ERROR] 파일이 존재하지 않습니다.")
        return None
    except Exception as e:
        print(f"[ERROR] 파일 해시 생성 중 오류 발생: {e}")
        return None

'''
테이블 데이터 추출 함수
camelot을 사용하여 PDF에서 표 데이터를 추출하고 텍스트로 변환
'''
def extract_tables_from_pdf(file_path):
    try:
        # camelot으로 테이블 추출 시도
        tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
        
        if len(tables) == 0:
            # lattice로 테이블이 발견되지 않으면 stream 방식 시도
            tables = camelot.read_pdf(file_path, pages='all', flavor='stream')
        
        table_docs = []
        
        for i, table in enumerate(tables):
            # 테이블을 DataFrame으로 변환
            df = table.df
            
            # 컬럼명과 데이터를 텍스트로 변환
            # 컬럼명에서 불필요한 공백 제거 및 정리
            header = [col.strip() for col in df.iloc[0]]
            
            # 테이블의 각 행을 처리
            for idx, row in df.iloc[1:].iterrows():
                row_data = [str(cell).strip() for cell in row]
                # 키-값 쌍 형태로 텍스트 구성 (컬럼명: 값)
                row_text = "\n".join([f"{header[i]}: {row_data[i]}" for i in range(len(header))])
                
                # 테이블 메타데이터 추가
                table_info = f"[테이블 {i+1}, 행 {idx}]\n{row_text}"
                table_docs.append(Document(page_content=table_info))
        
        return table_docs
    except Exception as e:
        print(f"[ERROR] 테이블 추출 중 오류 발생: {e}")
        return []

''' 
PDF에서 텍스트 추출 (PyMuPDFLoader + OCR + 테이블 추출 결합)
PyMuPDFLoader를 우선 사용 후 텍스트가 없는 페이지에만 pytesseract로 OCR 적용
추가로 camelot을 사용하여 테이블 데이터 추출
'''
def extract_text_from_pdf(file_path):
    try:
        # 일반 텍스트 추출
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
        # 텍스트가 적거나 없는 경우 OCR 보완
        if not docs or any(len(doc.page_content.strip()) < 100 for doc in docs):
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text or len(text.strip()) < 100:
                        img = page.to_image()
                        text = pytesseract.image_to_string(img, lang='kor+eng')
                    if text:
                        docs.append(Document(page_content=f"[페이지 {page_num+1}]\n{text}"))
        
        # 테이블 데이터 추출 및 추가
        table_docs = extract_tables_from_pdf(file_path)
        docs.extend(table_docs)
        
        return docs
    except Exception as e:
        print(f"[ERROR] PDF 텍스트 추출 중 오류 발생: {e}")
        return []


''' 
텍스트 분할 함수
" ", ". " 등을 넣어서 더 세밀하게 분할
테이블 데이터는 분할하지 않도록 조정
'''
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    # 테이블 데이터와 일반 텍스트 분리
    table_docs = [doc for doc in docs if doc.page_content.startswith("[테이블")]
    text_docs = [doc for doc in docs if not doc.page_content.startswith("[테이블")]
    
    # 일반 텍스트만 분할
    split_text_docs = text_splitter.split_documents(text_docs)
    
    # 테이블 데이터는 그대로 유지하여 합침
    return split_text_docs + table_docs

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
숫자 질문에 대한 패턴 인식 함수
개수, 몇 개, 몇 명 등의 패턴을 감지
'''
def is_count_question(query):
    patterns = [
        r'개수', r'몇\s*개', r'몇\s*명', r'총\s*개', r'총\s*수', 
        r'총\s*인원', r'수는', r'개는', r'몇\s*곳', r'몇\s*회', 
        r'얼마나', r'많은', r'있는', r'있습니까'
    ]
    return any(re.search(pattern, query) for pattern in patterns)

'''
RAG Pipeline 클래스
주어진 쿼리에 대해 PDF 파일에서 컨텍스트를 추출하고, 해당 컨텍스트와 함께 답변을 생성하는 전체 파이프라인
'''
class RAGPipeline:
    def __init__(self, generator):
        self.generator = generator

    def __call__(self, query, file):
        file_hash = get_file_hash(file)
        docs = extract_text_from_pdf(file.name)
        vectorstore = get_vectorstore(docs, file_hash)
        
        # MMR 검색을 통해 다양한 컨텍스트 확보
        retrieved_docs = vectorstore.max_marginal_relevance_search(query, k=7, fetch_k=20)
        
        # 테이블 관련 질문인 경우 테이블 데이터에 가중치 부여
        if is_count_question(query):
            # 테이블 데이터 문서 우선 배치
            table_docs = [doc for doc in retrieved_docs if "[테이블" in doc.page_content]
            other_docs = [doc for doc in retrieved_docs if "[테이블" not in doc.page_content]
            retrieved_docs = table_docs + other_docs
        
        formatted_context = format_context(retrieved_docs)
        return self.generator(query, formatted_context)

'''
Generator 클래스
Ollama API를 호출하여 사용자 쿼리와 컨텍스트를 기반으로 자연어 응답 생성
'''
class Generator:
    def __call__(self, query, context):
        # 숫자 질문인지 확인
        count_question = is_count_question(query)
        
        # 프롬프트 조정
        system_prompt = "당신은 정확한 문서 분석 AI 비서입니다. 제공된 컨텍스트를 기반으로 정확한 답변을 제공하세요."
        
        if count_question:
            system_prompt += " 특히 숫자나 개수를 묻는 질문에는 정확한 숫자로 답변하고, 계산 과정을 설명해 주세요."
        
        formatted_prompt = f"""
        {system_prompt}
        
        질문: {query}

        컨텍스트:
        {context}
        
        컨텍스트에서 찾은 정보를 바탕으로 명확하고 간결하게 답변해 주세요.
        """
        
        if count_question:
            formatted_prompt += """
            이 질문은 개수나 숫자를 묻고 있습니다. 컨텍스트에서 관련 항목들을 찾아 정확히 계산해 주세요.
            예를 들어 '지역이 부산광역시인 기업의 수'를 묻는다면, 표에서 지역이 '부산광역시'로 표시된 기업들의 개수를 세어 정확한 숫자를 답변해야 합니다.
            """

        response = ollama.chat(
            model='benedict/linkbricks-llama3.1-korean:8b',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt}
            ]
        )
        return response['message']['content']

# Gradio ChatInterface로 UI 설정
generator = Generator()
rag_pipeline = RAGPipeline(generator)

chatbot = gr.ChatInterface(
    fn=lambda msg, hist, file: rag_pipeline(msg, file),
    title="[benedict/linkbricks-llama3.1-korean:8b] Optimized RAG System with Table Support",
    description="PDF 파일을 업로드하고 질문하세요. 시스템이 관련 컨텍스트를 검색하고 표 데이터를 포함한 응답을 생성합니다.",
    additional_inputs=[gr.File(label="📄 PDF 파일", file_types=[".pdf"])]
)

chatbot.launch()
