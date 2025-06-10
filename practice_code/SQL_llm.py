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
import json

# Cache for retrievers and DataFrames
retriever_cache = {}
dataframe_cache = {}

# Embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

QA_LOG_FILE = "qa_data.json" 

def save_qa_to_json(user_question, assistant_answer, system_prompt="당신은 AI 기업 정보를 기반으로 분석해주는 지식 비서입니다."):
    """QA 로그를 JSON 파일에 저장"""
    qa_entry = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": assistant_answer}
        ]
    }

    if os.path.exists(QA_LOG_FILE):
        with open(QA_LOG_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(qa_entry)

    with open(QA_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"💾 QA 저장 완료: {user_question}")

def get_file_hash(file):
    """파일 해시 생성"""
    try:
        file_path = file.name if hasattr(file, "name") else file
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"[ERROR] 해시 생성 실패: {e}")
        return None

def is_sql_query(query):
    """SQL 쿼리 패턴 감지"""
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
        r'.*지역.*몇',
        r'시각지능.*몇',
        r'분석지능.*몇',
        r'언어.*음성.*몇',
        r'행동지능.*몇',
        r'전체.*몇',
        r'총.*개',
        r'설립연도.*몇'
    ]
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in sql_patterns)

def convert_to_sql(query):
    """자연어를 SQL 쿼리로 변환"""
    query_lower = query.lower()
    
    # 지역별 기업 수 쿼리
    regions = {
        '부산': "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%부산%'",
        '서울': "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%서울%'",
        '광주': "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%광주%'",
        '인천': "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%인천%'",
        '대전': "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%대전%'",
        '대구': "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%대구%'",
        '울산': "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%울산%'",
        '경기': "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%경기%'",
        '강원': "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%강원%'",
        '제주': "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%제주%'",
    }
    
    for region, sql in regions.items():
        if region in query and ('몇' in query or '개수' in query or 'count' in query_lower):
            return sql
    
    # 경상도 관련
    if ('경남' in query or '경상남도' in query) and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%경상남도%'"
    
    if ('경북' in query or '경상북도' in query) and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%경상북도%'"
    
    if ('경상' in query or '경상도' in query) and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE (지역 LIKE '%경상남도%' OR 지역 LIKE '%경상북도%')"
    
    # 충청도 관련
    if ('충남' in query or '충청남도' in query) and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%충청남도%'"
    
    if ('충북' in query or '충청북도' in query) and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%충청북도%'"
    
    if '충청' in query and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE (지역 LIKE '%충청남도%' OR 지역 LIKE '%충청북도%')"
    
    # 전라도 관련
    if ('전남' in query or '전라남도' in query) and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%전라남도%'"
    
    if ('전북' in query or '전라북도' in query) and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 지역 LIKE '%전라북도%'"
    
    if '전라도' in query and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE (지역 LIKE '%전라남도%' OR 지역 LIKE '%전라북도%')"
    
    # 전문분야별 개수
    if '시각지능' in query and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 전문분야 LIKE '%시각지능%'"
    
    if '분석지능' in query and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 전문분야 LIKE '%분석지능%'"
    
    if ('언어·음성지능' in query or '언어,음성지능' in query or '언어음성지능' in query) and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 전문분야 LIKE '%언어·음성지능%'"
    
    if '행동지능' in query and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업수 FROM companies WHERE 전문분야 LIKE '%행동지능%'"
    
    # 설립연도별 분석
    if '부산' in query and ('설립연도' in query or '연도별' in query):
        return "SELECT SUBSTR(회사설립일, 1, 4) AS 설립연도, COUNT(*) AS 기업수 FROM companies WHERE 지역 LIKE '%부산%' GROUP BY 설립연도 ORDER BY 설립연도"
    
    if '서울' in query and ('설립연도' in query or '연도별' in query):
        return "SELECT SUBSTR(회사설립일, 1, 4) AS 설립연도, COUNT(*) AS 기업수 FROM companies WHERE 지역 LIKE '%서울%' GROUP BY 설립연도 ORDER BY 설립연도"
    
    if ('전체' in query or '모든' in query) and ('설립연도' in query or '연도별' in query):
        return "SELECT SUBSTR(회사설립일, 1, 4) AS 설립연도, COUNT(*) AS 기업수 FROM companies GROUP BY 설립연도 ORDER BY 설립연도"
    
    # 전체 기업 수
    if ('전체' in query or '총' in query or '모든' in query) and ('몇' in query or '개수' in query):
        return "SELECT COUNT(*) as 기업명 FROM companies"
    
    return None

def create_sqlite_from_dataframe(df, db_path="temp_companies.db"):
    """DataFrame을 SQLite DB로 변환"""
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql('companies', conn, if_exists='replace', index=False)
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] SQLite DB 생성 실패: {e}")
        return False

def execute_sql_query(sql_query, db_path="temp_companies.db"):
    """SQL 쿼리 실행"""
    try:
        conn = sqlite3.connect(db_path)
        result = pd.read_sql_query(sql_query, conn)
        conn.close()
        return result
    except Exception as e:
        print(f"[ERROR] SQL 쿼리 실행 실패: {e}")
        return None

def extract_tables_from_pdf(file_path, save_csv_path="auto_extracted_table.csv"):
    """PDF에서 테이블 추출"""
    try:
        tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
        if len(tables) == 0:
            tables = camelot.read_pdf(file_path, pages='all', flavor='stream')
        
        table_docs = []
        all_dfs = []
        
        for i, table in enumerate(tables):
            df = table.df
            all_dfs.append(df)
            
            if len(df) > 0:
                header = [str(col).strip() for col in df.iloc[0]]
                df.columns = header
                df = df.iloc[1:].reset_index(drop=True)
                
                for idx, row in df.iterrows():
                    row_data = [str(cell).strip() for cell in row]
                    row_text = "\n".join([f"{header[i]}: {row_data[i]}" for i in range(len(header)) if i < len(row_data)])
                    table_docs.append(Document(page_content=f"[테이블 {i+1}, 행 {idx+1}]\n{row_text}"))
        
        if all_dfs:
            main_df = all_dfs[0].copy()
            if len(main_df) > 0:
                header = [str(col).strip() for col in main_df.iloc[0]]
                main_df.columns = header
                main_df = main_df.iloc[1:].reset_index(drop=True)
                
                for df in all_dfs[1:]:
                    if len(df) > 0:
                        df.columns = header[:len(df.columns)]
                        df = df.iloc[1:].reset_index(drop=True)
                        main_df = pd.concat([main_df, df], ignore_index=True)
                
                main_df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
                print(f"✅ CSV로 테이블 저장 완료: {save_csv_path}")
                
                dataframe_cache['companies'] = main_df
                create_sqlite_from_dataframe(main_df)
                
        return table_docs
    except Exception as e:
        print(f"[ERROR] 테이블 추출 실패: {e}")
        return []

def extract_text_from_pdf(file_path):
    """PDF에서 텍스트 추출"""
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
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
        
        table_docs = extract_tables_from_pdf(file_path)
        docs.extend(table_docs)
        
        return docs
    except Exception as e:
        print(f"[ERROR] PDF 텍스트 추출 실패: {e}")
        return []

def split_text(docs):
    """텍스트 분할"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    table_docs = [doc for doc in docs if doc.page_content.startswith("[테이블")]
    text_docs = [doc for doc in docs if not doc.page_content.startswith("[테이블")]
    
    split_text_docs = splitter.split_documents(text_docs)
    return split_text_docs + table_docs

def format_context(retrieved_docs, max_length=3000):
    """컨텍스트 포맷팅"""
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return context[:max_length]

def is_count_question(query):
    """개수 질문 패턴 감지"""
    patterns = [
        r'개수', r'몇\s*개', r'몇\s*명', r'총\s*개', r'총\s*수', 
        r'총\s*인원', r'수는', r'개는', r'몇\s*곳', r'몇\s*회', 
        r'얼마나', r'많은', r'있는', r'있습니까'
    ]
    return any(re.search(pattern, query) for pattern in patterns)

def format_sql_result_for_llm(sql_result, query):
    """SQL 결과를 LLM이 이해하기 쉽게 포맷팅"""
    if sql_result is None or len(sql_result) == 0:
        return "검색 결과가 없습니다."
    
    # 단일 개수 결과
    if len(sql_result.columns) == 1 and len(sql_result) == 1:
        count = sql_result.iloc[0, 0]
        return f"검색 결과: {count}개"
    
    # 연도별 또는 그룹별 결과
    if len(sql_result.columns) == 2:
        result_text = "검색 결과:\n"
        for _, row in sql_result.iterrows():
            result_text += f"- {row.iloc[0]}: {row.iloc[1]}개\n"
        total = sql_result.iloc[:, 1].sum()
        result_text += f"총합: {total}개"
        return result_text
    
    return sql_result.to_string()

class Generator:
    def __call__(self, query, context=None, sql_result=None):
        """LLM을 사용한 답변 생성"""
        system_prompt = "당신은 정확한 문서 분석 AI 비서입니다. 제공된 정보를 바탕으로 자연스럽고 정확한 답변을 제공하세요."
        
        prompt = f"질문: {query}\n\n"
        
        # SQL 결과가 있는 경우 우선 사용
        if sql_result is not None:
            formatted_result = format_sql_result_for_llm(sql_result, query)
            prompt += f"데이터베이스 검색 결과:\n{formatted_result}\n\n"
            prompt += "위의 검색 결과를 바탕으로 질문에 대해 자연스럽고 정확하게 답변해 주세요."
        elif context:
            prompt += f"관련 문서 내용:\n{context}\n\n"
            prompt += "위의 문서 내용을 바탕으로 질문에 대해 답변해 주세요."
        
        try:
            response = ollama.chat(
                model='benedict/linkbricks-llama3.1-korean:8b ',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            # SQL 결과가 있으면 간단한 템플릿 답변
            if sql_result is not None:
                if len(sql_result) == 1 and len(sql_result.columns) == 1:
                    count = sql_result.iloc[0, 0]
                    return f"검색하신 조건에 해당하는 기업 수는 **{count}개**입니다."
                else:
                    return f"검색 결과:\n{sql_result.to_string()}"
            
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

class RAGPipeline:
    def __init__(self, generator):
        self.generator = generator

    def __call__(self, query, file):
        """메인 RAG 파이프라인"""
        if not file:
            return "PDF 파일을 업로드해 주세요."
        
        file_hash = get_file_hash(file)
        
        # 먼저 데이터 추출 (캐시 확인)
        if 'companies' not in dataframe_cache:
            print("📊 PDF에서 데이터 추출 중...")
            docs = extract_text_from_pdf(file.name)
        
        # SQL 쿼리로 처리 가능한지 확인
        if is_sql_query(query):
            print("🔍 SQL 쿼리로 처리 가능한 질문 감지")
            
            sql_query = convert_to_sql(query)
            if sql_query:
                print(f"📝 생성된 SQL: {sql_query}")
                sql_result = execute_sql_query(sql_query)
                
                if sql_result is not None:
                    print(f"✅ SQL 결과 획득")
                    # SQL 결과를 LLM으로 자연스럽게 답변 생성
                    response = self.generator(query, sql_result=sql_result)
                    return response
                else:
                    print("❌ SQL 쿼리 실행 실패, RAG 검색으로 전환")
        
        # 일반 RAG 검색
        print("🔎 일반 RAG 검색 실행")
        
        if file_hash in retriever_cache:
            print("📦 캐시된 retriever 사용 중...")
            vectorstore = retriever_cache[file_hash]
        else:
            print("🔄 새로운 벡터스토어 생성 중...")
            if 'companies' not in dataframe_cache:
                docs = extract_text_from_pdf(file.name)
            else:
                docs = extract_text_from_pdf(file.name)
            
            split_docs = split_text(docs)
            vectorstore = Chroma.from_documents(split_docs, embeddings)
            retriever_cache[file_hash] = vectorstore

        retrieved_docs = vectorstore.max_marginal_relevance_search(query, k=7, fetch_k=20)
        
        if is_count_question(query):
            table_docs = [doc for doc in retrieved_docs if "[테이블" in doc.page_content]
            other_docs = [doc for doc in retrieved_docs if "[테이블" not in doc.page_content]
            retrieved_docs = table_docs + other_docs

        context = format_context(retrieved_docs)
        return self.generator(query, context=context)

# Gradio 인터페이스
generator = Generator()
rag_pipeline = RAGPipeline(generator)

def chat_interface(message, history, file):
    """채팅 인터페이스 함수"""
    try:
        response = rag_pipeline(message, file)
        
        # QA 로그 저장
        save_qa_to_json(message, response)
        
        return response
    except Exception as e:
        error_msg = f"처리 중 오류가 발생했습니다: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg

# Gradio 애플리케이션
chatbot = gr.ChatInterface(
    fn=chat_interface,
    title="🤖 통합 PDF RAG+SQL 분석 시스템",
    description="""
    📊 **PDF 파일을 업로드하고 질문하세요!**
    
    💡 **지원하는 질문 유형:**
    - 🏢 **지역별 기업 수**: "부산에 있는 기업이 몇 개야?", "서울 기업 수는?"
    - 🎯 **전문분야별 분석**: "시각지능 분야 기업이 몇 개야?", "분석지능 기업 수는?"
    - 📅 **연도별 분석**: "부산 기업들의 설립연도별 분포는?", "전체 설립연도별 현황은?"
    - 📈 **전체 통계**: "전체 기업 수는?", "총 몇 개 기업이 있어?"
    - 💬 **일반 질문**: 문서 내용에 대한 자유로운 질문
    
    ⚡ **특징:**
    - 구조화된 데이터는 SQL로 빠르게 처리
    - 복잡한 질문은 AI 검색으로 대응
    - 자연스러운 한국어 답변 제공
    """,
    additional_inputs=[
        gr.File(
            label="📄 PDF 파일 업로드", 
            file_types=[".pdf"],
            type="filepath"
        )
    ],
    theme="soft"
)

chatbot.launch()
