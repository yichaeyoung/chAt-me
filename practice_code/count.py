import gradio as gr
import pandas as pd
import hashlib
import os
import re
import io
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.document_loaders import PyMuPDFLoader
import camelot
import ollama
import sqlite3
from datetime import datetime

# Cache for retrievers and DataFrames
retriever_cache = {}
dataframe_cache = {}

# Embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# File hash for caching
def get_file_hash(file):
    try:
        file_path = file.name if hasattr(file, "name") else file
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"[ERROR] 해시 생성 실패: {e}")
        return None

# SQL 쿼리 패턴 감지
def is_sql_query(query):
    sql_patterns = [
        r'select\s+.*\s+from',
        r'count\(\*\)',
        r'where\s+.*\s+like',
        r'group\s+by',
        r'order\s+by',
        r'부산.*몇\s*개',
        r'서울.*몇\s*개',
        r'광주.*몇\s*개',
        r'경상.*몇\s*개',
        r'경남.*몇\s*개',
        r'경북.*몇\s*개',
        r'인천.*몇\s*개',
        r'제주.*몇\s*개',
        r'강원.*몇\s*개',
        r'대전.*몇\s*개',
        r'경기.*몇\s*개',
        r'충남.*몇\s*개',
        r'충북.*몇\s*개',
        r'충청.*몇\s*개',
        r'대구.*몇\s*개',
        r'전라.*몇\s*개',
        r'전남.*몇\s*개',
        r'전북.*몇\s*개',
        r'울산.*몇\s*개',
        r'.*지역.*개수',
        r'.*지역.*몇'
    ]
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in sql_patterns)

# 자연어를 SQL 쿼리로 변환
def convert_to_sql(query):
    query_lower = query.lower()
    
    if '부산' in query and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%부산%'"
    
    if '서울' in query and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%서울%'"
    
    if '광주' in query and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%광주%'"
    
    if ('경남' in query or '경상남도' in query) and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%경상남도%'"
    
    if ('경북' in query or '경상북도' in query) and ('몇'in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%경상북도%'"

    if ('경상' in query or '경상도' in query) and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE (지역 LIKE '%경상남도%' OR 지역 LIKE '%경상북도%')"

    if '인천' in query and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%인천%'"

    if ('제주' in query or '제주도' in query) and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%제주%'"

    if ('강원' in query or '강원도' in query) and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%강원%'"

    if '대전' in query and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%대전%'"

    if '경기' in query and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%경기%'"

    if ('충남' in query or '충청남도' in query) and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%충청남도%'"

    if ('충북' in query or '충청북도' in query) and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%충청북도%'"

    if '충청' in query and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE (지역 LIKE '%충청남도%' OR 지역 LIKE '%충청북도%')"

    if '대구' in query and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%대구광역시%'"

    if '전라도' in query and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE (지역 LIKE '%전라남도%' OR 지역 LIKE '%전라북도%')"

    if ('전남' in query or '전라남도' in query) and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%전라남도%'"

    if ('전북' in query or '전라북도' in query) and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%전라북도%'"

    if '울산' in query and ('몇' in query or '개수' in query or 'count' in query_lower):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%울산광역시%'"
    
    if '부산' in query and ('설립연도' in query or '회사설립일' in query or '연도별' in query or '년도별' in query):
        return "SELECT SUBSTR(회사설립일, 1, 4) AS 설립연도, count(*) FROM companies WHERE 지역 LIKE '%부산%' GROUP BY 설립연도 ORDER BY 설립연도;"
    
    if '서울' in query and ('설립연도' in query or '회사설립일' in query or '연도별' in query or '년도별' in query):
        return "SELECT SUBSTR(회사설립일, 1, 4) AS 설립연도, count(*) FROM companies WHERE 지역 LIKE '%서울%' GROUP BY 설립연도 ORDER BY 설립연도;"
    
    if ('전체' in query or '모든 지역' in query or '지역별' in query) and ('연도별' in query or '설립연도' in query):
        return """
        SELECT SUBSTR(회사설립일, 1, 4) AS 설립연도, COUNT(*) AS 기업수 
        FROM companies 
        GROUP BY 설립연도 
        ORDER BY 설립연도
        """
    
    # 지역별 개수 쿼리
    region_match = re.search(r'(\w+광역시|\w+시|\w+도|\w+특별시).*(?:몇|개수|count)', query)
    if region_match:
        region = region_match.group(1)
        return f"SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%{region}%'"
    
    # 전문분야별 개수 쿼리
    if '시각지능' in query and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 전문분야 LIKE '%시각지능%'"
    
    # 기본 전체 개수
    if '전체' in query and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 총기업수 FROM companies"
    
    return None

# DataFrame을 SQLite DB로 변환
def create_sqlite_from_dataframe(df, db_path="temp_companies.db"):
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql('companies', conn, if_exists='replace', index=False)
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] SQLite DB 생성 실패: {e}")
        return False

# SQL 쿼리 실행
def execute_sql_query(sql_query, db_path="temp_companies.db"):
    try:
        conn = sqlite3.connect(db_path)
        result = pd.read_sql_query(sql_query, conn)
        conn.close()
        return result
    except Exception as e:
        print(f"[ERROR] SQL 쿼리 실행 실패: {e}")
        return None

# OCR + Text + Table extraction from PDF
def extract_tables_from_pdf(file_path, save_csv_path="auto_extracted_table.csv"):
    try:
        tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
        if len(tables) == 0:
            tables = camelot.read_pdf(file_path, pages='all', flavor='stream')
        
        table_docs = []
        all_dfs = []
        
        for i, table in enumerate(tables):
            df = table.df
            all_dfs.append(df)
            
            # 테이블 구조 분석 및 헤더 설정
            if len(df) > 0:
                # 첫 번째 행을 헤더로 사용
                header = [str(col).strip() for col in df.iloc[0]]
                df.columns = header
                df = df.iloc[1:].reset_index(drop=True)
                
                # 각 행을 문서로 변환
                for idx, row in df.iterrows():
                    row_data = [str(cell).strip() for cell in row]
                    row_text = "\n".join([f"{header[i]}: {row_data[i]}" for i in range(len(header)) if i < len(row_data)])
                    table_docs.append(Document(page_content=f"[테이블 {i+1}, 행 {idx+1}]\n{row_text}"))
        
        # 모든 테이블을 하나의 DataFrame으로 결합
        if all_dfs:
            # 첫 번째 테이블의 헤더를 기준으로 설정
            main_df = all_dfs[0].copy()
            if len(main_df) > 0:
                header = [str(col).strip() for col in main_df.iloc[0]]
                main_df.columns = header
                main_df = main_df.iloc[1:].reset_index(drop=True)
                
                # 나머지 테이블들 추가
                for df in all_dfs[1:]:
                    if len(df) > 0:
                        df.columns = header[:len(df.columns)]
                        df = df.iloc[1:].reset_index(drop=True)
                        main_df = pd.concat([main_df, df], ignore_index=True)
                
                # CSV 저장
                main_df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
                print(f"✅ CSV로 테이블 저장 완료: {save_csv_path}")
                
                # DataFrame 캐시에 저장
                dataframe_cache['companies'] = main_df
                
                # SQLite DB 생성
                create_sqlite_from_dataframe(main_df)
                
        return table_docs
    except Exception as e:
        print(f"[ERROR] 테이블 추출 실패: {e}")
        return []

def extract_text_from_pdf(file_path):
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
        # 텍스트 품질 확인 및 OCR 보완
        if not docs or any(len(doc.page_content.strip()) < 100 for doc in docs):
            print("📄 OCR을 통한 텍스트 추출 시작...")
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text or len(text.strip()) < 100:
                        img = page.to_image()
                        text = pytesseract.image_to_string(img, lang='kor+eng')
                    if text:
                        docs.append(Document(page_content=f"[페이지 {page_num+1}]\n{text}"))
        
        # 테이블 추출 및 추가
        table_docs = extract_tables_from_pdf(file_path)
        docs.extend(table_docs)
        
        return docs
    except Exception as e:
        print(f"[ERROR] PDF 텍스트 추출 실패: {e}")
        return []

def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    # 테이블과 텍스트 문서 분리
    table_docs = [doc for doc in docs if doc.page_content.startswith("[테이블")]
    text_docs = [doc for doc in docs if not doc.page_content.startswith("[테이블")]
    
    # 텍스트 문서만 분할
    split_text_docs = splitter.split_documents(text_docs)
    
    return split_text_docs + table_docs

def format_context(retrieved_docs, max_length=3000):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return context[:max_length]

def is_count_question(query):
    patterns = [
        r'개수', r'몇\s*개', r'몇\s*명', r'총\s*개', r'총\s*수', 
        r'총\s*인원', r'수는', r'개는', r'몇\s*곳', r'몇\s*회', 
        r'얼마나', r'많은', r'있는', r'있습니까'
    ]
    return any(re.search(pattern, query) for pattern in patterns)

class Generator:
    def __call__(self, query, context, sql_result=None):
        count_q = is_count_question(query)
        system_prompt = "당신은 정확한 문서 분석 AI 비서입니다. 제공된 컨텍스트를 기반으로 정확한 답변을 제공하세요."
        
        if count_q:
            system_prompt += " 특히 숫자나 개수를 묻는 질문에는 정확한 숫자로 답변하고, 계산 과정을 설명해 주세요."

        prompt = f"""
        {system_prompt}

        질문: {query}

        컨텍스트:
        {context}
        """
        
        # SQL 결과가 있는 경우 추가
        if sql_result is not None:
            prompt += f"""
            
        SQL 쿼리 결과:
        {sql_result.to_string()}
        
        위의 SQL 쿼리 결과를 바탕으로 정확한 답변을 제공해 주세요.
        """

        if count_q:
            prompt += """
            
        이 질문은 개수나 숫자를 묻고 있습니다. SQL 결과나 관련 항목들을 찾아 정확히 계산해 주세요.
        """

        try:
            response = ollama.chat(
                model='benedict/linkbricks-llama3.1-korean:8b',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

class RAGPipeline:
    def __init__(self, generator):
        self.generator = generator

    def __call__(self, query, file):
        if not file:
            return "PDF 파일을 업로드해 주세요."
        
        file_hash = get_file_hash(file)
        
        # SQL 쿼리 감지 및 처리
        if is_sql_query(query):
            print("🔍 SQL 쿼리 감지됨")
            
            # 캐시된 DataFrame이 없으면 PDF에서 추출
            if 'companies' not in dataframe_cache:
                print("📊 PDF에서 데이터 추출 중...")
                docs = extract_text_from_pdf(file.name)
            
            # SQL 쿼리 생성 및 실행
            sql_query = convert_to_sql(query)
            if sql_query:
                print(f"📝 생성된 SQL: {sql_query}")
                sql_result = execute_sql_query(sql_query)
                
                if sql_result is not None:
                    print(f"✅ SQL 결과: {sql_result}")
                    return f"SQL 쿼리 실행 결과:\n\n{sql_result.to_string()}\n\n질문하신 내용에 대한 답변: {sql_result.iloc[0, 0]}개"
                else:
                    return "SQL 쿼리 실행 중 오류가 발생했습니다."
            else:
                return "해당 질문을 SQL 쿼리로 변환할 수 없습니다. 일반 검색으로 진행합니다."
        
        # 일반 RAG 검색
        print("🔎 일반 RAG 검색 실행")
        
        # 캐시된 retriever 확인
        if file_hash in retriever_cache:
            print("📦 캐시된 retriever 사용 중...")
            vectorstore = retriever_cache[file_hash]
        else:
            print("🔄 새로운 벡터스토어 생성 중...")
            docs = extract_text_from_pdf(file.name)
            split_docs = split_text(docs)
            vectorstore = Chroma.from_documents(split_docs, embeddings)
            retriever_cache[file_hash] = vectorstore

        # 문서 검색
        retrieved_docs = vectorstore.max_marginal_relevance_search(query, k=7, fetch_k=20)
        
        # 개수 질문의 경우 테이블 우선 처리
        if is_count_question(query):
            table_docs = [doc for doc in retrieved_docs if "[테이블" in doc.page_content]
            other_docs = [doc for doc in retrieved_docs if "[테이블" not in doc.page_content]
            retrieved_docs = table_docs + other_docs

        context = format_context(retrieved_docs)
        return self.generator(query, context)

# Gradio Chat Interface
generator = Generator()
rag_pipeline = RAGPipeline(generator)

def chat_interface(message, history, file):
    try:
        response = rag_pipeline(message, file)
        return response
    except Exception as e:
        return f"처리 중 오류가 발생했습니다: {str(e)}"

# Gradio 인터페이스 설정
chatbot = gr.ChatInterface(
    fn=chat_interface,
    title="[LLM] PDF Table RAG+SQL 통합 시스템",
    description="""
    📊 PDF 파일을 업로드하고 질문하세요. 
    
    💡 **예시 질문:**
    - "부산광역시에 위치한 기업의 수는 몇개야?"
    - "시각지능 분야 기업이 몇개야?"
    - "전체 기업 수는?"
    """,
    additional_inputs=[
        gr.File(
            label="📄 PDF 파일", 
            file_types=[".pdf"],
            type="filepath"
        )
    ]
)

if __name__ == "__main__":
    chatbot.launch(share=True, debug=True)
