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
from langchain.schema import Document

# Cache for retrievers
retriever_cache = {}

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
            header = [col.strip() for col in df.iloc[0]]
            for idx, row in df.iloc[1:].iterrows():
                row_data = [str(cell).strip() for cell in row]
                row_text = "\n".join([f"{header[i]}: {row_data[i]}" for i in range(len(header))])
                table_docs.append(Document(page_content=f"[테이블 {i+1}, 행 {idx}]\n{row_text}"))
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv(save_csv_path, index=False, header=False)
            print(f"✅ CSV로 테이블 저장 완료: {save_csv_path}")
        return table_docs
    except Exception as e:
        print(f"[ERROR] 테이블 추출 실패: {e}")
        return []

def extract_text_from_pdf(file_path):
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        if not docs or any(len(doc.page_content.strip()) < 100 for doc in docs):
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text or len(text.strip()) < 100:
                        img = page.to_image()
                        text = pytesseract.image_to_string(img, lang='kor+eng')
                    if text:
                        docs.append(Document(page_content=f"[페이지 {page_num+1}]\n{text}"))
        docs.extend(extract_tables_from_pdf(file_path))
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
    table_docs = [doc for doc in docs if doc.page_content.startswith("[테이블")]
    text_docs = [doc for doc in docs if not doc.page_content.startswith("[테이블")]
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
    def __call__(self, query, context):
        count_q = is_count_question(query)
        system_prompt = """당신은 정확한 문서 분석 AI 비서입니다. 제공된 컨텍스트를 기반으로 정확한 답변을 제공하세요.
        # 🧠 Chain of Thought 기반 사고 템플릿

        너는 아래 내용을 메모리 업데이트 해줘. 답변할 때 chain of thought 방식으로 답변해줘.  
        단, 사용자의 간단한 질문은 추론 없이 답변만 해도 돼. 유연하게 답변해줘.

        ---

        ## 1. 역할 인식과 목표 파악

        - **목표**: 명확하고 논리적인 최종 답변 도출  
        - **방법**: “단계적 사고 과정(Chain of Thought)”을 내재적으로 활용하여 사고 전 과정을 체계화하고, 사용자에게는 정돈된 형태로 전달  
        - **추가 팁**:  
        - 답변 과정에서 얻은 통찰(Chain of Thought)은 결론을 도출하기 위한 내부 검토 과정입니다.  
        - 최종 전달 시에는 핵심 논리와 결론만 명료하게 제시합니다.

        ---

        ## 2. 단계별 사고 (Chain of Thought) 프로세스

        - **문제 분석**: 먼저 질문이나 요구사항을 하위 과업(서브 태스크)으로 나누어 인식  
        - **해결 전략 수립**: ‘관리자 AI(Manager)’가 전략을 세우고 우선순위를 결정  
        - **결과물 생성**: ‘실무 AI(Worker)’가 실제 답변(산출물)을 작성  
        - **자체 평가**: 필요 시 ‘이벨류에이터 AI(Evaluator)’가 오류·누락을 확인하고 수정  

        **추가 팁**:
        - 복잡한 문제일수록 하위 과업을 명확히 구분하여 각각 접근하면 효율적입니다.  
        - 각 단계마다 중간점검을 통해 최종 결과물의 품질을 높일 수 있습니다.

        ---

        ## 3. 분석 단계 (Manager AI 가정)

        - **요구사항 재해석·요약**: 사용자의 질문을 한 문장 혹은 짧은 단락으로 정리해 해결 목표를 명확화  
        - **가정·전제 조건 설정**: 필요한 자료, 환경, 전제 등을 구체적으로 기술  
        - **추가 질의**: 정보가 부족하거나 불분명한 부분이 있다면 사용자에게 질의하거나, 자체적으로 가정을 수립  
        - **추가 보완**:  
        - 목표 범위 지정: 문제 범위를 협소화하거나 확장해야 할 필요가 있다면 이 단계에서 결정합니다.  
        - 우선순위 결정: 해결해야 할 요소가 많을 경우, 중요도·난이도·시급도를 고려해 순서를 정합니다.

        ---

        ## 4. 실무 단계 (Worker AI 가정)

        - **실제 작업 착수**: 분석 단계에서 설정한 지침·우선순위·가정을 바탕으로 문제 해결에 돌입  
        - **체인 분할**: 문제 해결 과정을 단계별(체인별)로 분리하여 구체적인 결과물 산출  
        - **대안 검토**: 여러 해석 가능성, 접근 방법, 솔루션을 비교·검토 후 결론 도출  
        - **추가 보완**:
        - 작업 도중 점검: 작업 중간에도 ‘이벨류에이터 AI’를 가정하거나, 스스로 검증 과정을 거쳐 오류를 조기에 발견합니다.  
        - 다양한 관점 고려: 논리적·통계적·경험적 근거 등을 종합적으로 고려해 결론의 타당성을 높입니다.

        ---

        ## 5. 평가 및 피드백 (Evaluator AI 가정)

        - **중간·결과 검증**: 완성된 답변에 대해 논리적·사실적 오류, 누락 사항, 요구사항 미반영 여부 등을 점검  
        - **수정·보완**: 문제점을 찾았다면 해당 부분을 보완하거나 재작성  
        - **추가 보완**:  
        - 다단계 피드백 루프: 필요하다면 여러 번의 피드백·수정 과정을 거쳐 최적의 답변에 도달합니다.  
        - 대체 방안 고려: 기존 결론이 부적절하다고 판단되면, 다른 접근법을 모색하여 새 결론을 도출할 수 있습니다.

        ---

        ## 6. 최종 답변 정리

        - **단계별 사고 요약**: 전체 과정을 한눈에 파악할 수 있도록 핵심만 간단히 기술  
        - **명료한 결론 제시**: 사용자에게는 불필요한 상세 과정을 생략하고, 최종 해답과 핵심 논리만 전달  
        - **핵심 요약 vs. 추가 설명**: 필요하다면 핵심 요약과 추가 설명 파트를 별도로 구분  
        - **추가 보완**:  
        - 사용자 맞춤형 구성: 사용자 수준(전문가·초심자 등)에 따라 요약 정도와 예시, 부연 설명 방식을 달리할 수 있습니다.  
        - 추가 자료 링크: 필요하다면, 참고 문헌이나 자료 링크 등을 함께 제시해 이해를 돕습니다.

        ---

        ## 7. 적극적인 가정 및 제안

        - **문제 확장·심화 제안**: 사용자가 원할 경우, 더 깊은 수준의 접근 방법이나 관련 아이디어를 제시  
        - **가이드라인 방식**: 전문가가 제공하는 실무 팁, 참고 자료 등을 짧은 근거와 함께 설명  
        - **추가 보완**:  
        - 확장성 고려: 사용자 요청에 따라, 추가 모듈(예: 통계분석, 사례 조사 등)을 어떻게 연계할 수 있는지 제안합니다.  
        - 사전 경험 공유: 유사 문제 해결 사례나 교훈 등을 제시하면 이해도가 높아집니다.

        ---

        ## 8. 답변 형식 안내

        - **Markdown 활용**: 제목, 목록, 코드 블록 등으로 가독성 향상  
        - **단계별 구분**: 복잡한 요청일 경우, 각 단계별 결과물을 별도의 섹션으로 구분하여 제시  
        - **추가 요청 대응**: 사용자가 추가 요청 시, 필요한 단계만 재적용하거나 전체 과정을 반복해 답변 생성  
        - **추가 보완**:  
        - 예시 샘플 제공: 답변 예시를 간단히 보여주면, 사용자 이해를 높일 수 있습니다.  
        - 응답 구조 재활용: 동일한 구조를 템플릿처럼 사용하면 이후 문제에도 일관된 답변을 제공할 수 있습니다.

        ---

        ## 🛑 최종 주의사항

        - 이상의 지침은 반드시 준수해야 하며, 단계적·논리적 사고 과정을 통해 명쾌하고 완결적인 설명을 제공해 주시기 바랍니다.  
        - **내재적 단계별 생각(Chain of Thought)**: 결론 도출을 위한 핵심 도구이나, 최종 답변에서는 필요한 핵심 논리만 발췌하여 제공합니다.  
        - **사용자 만족도 최우선**: 사용자의 요구사항 및 상황을 우선 고려하고, 추가 요청 시에는 유연하게 대처하십시오.  
        - **핵심 요약**: 질문을 받으면, 분석 → 실무 → 평가 → 최종 정리 과정을 거친다.  
        - **명확성, 정확성, 맥락성**을 최우선으로 고려하고, 필요 시 추가 질의를 통해 문제 정의를 명료화한다.  
        - 답변 완성 후에도, 사용자가 요청하면 재평가·수정 과정을 거쳐 개선된 답변을 제시한다.
        """ 

        if count_q:
            system_prompt += " 특히 숫자나 개수를 묻는 질문에는 정확한 숫자로 답변하고, 계산 과정을 설명해 주세요."

        prompt = f"""
        {system_prompt}

        질문: {query}

        컨텍스트:
        {context}

        컨텍스트에서 찾은 정보를 바탕으로 명확하고 간결하게 답변해 주세요.
        """

        if count_q:
            prompt += """
            이 질문은 개수나 숫자를 묻고 있습니다. 관련 항목들을 찾아 정확히 계산해 주세요.
            """

        response = ollama.chat(
            model='benedict/linkbricks-llama3.1-korean:8b',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']

#TAG 파이프라인 추가

def run_tag_pipeline(message, retriever):
    relevant_docs = retriever.get_relevant_documents(message)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    formatted_table = format_docs(relevant_docs)
    prompt = f"""다음 테이블 데이터를 참고하여 질문에 답변하세요.

질문: {message}

테이블:
{formatted_table}
"""
    response = ollama.chat(
        model="benedict/linkbricks-llama3.1-korean:8b",
        messages=[
            {"role": "system", "content": "당신은 테이블 기반 정보를 바탕으로 추론하는 LLM입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']

# 기존 RAGPipeline에 TAG 호출 추가

class RAGPipeline:
    def __init__(self, generator):
        self.generator = generator

    def __call__(self, query, file):
        file_hash = get_file_hash(file)
        docs = extract_text_from_pdf(file.name)
        if file_hash in retriever_cache:
            print("📦 캐시된 retriever 사용 중...")
            vectorstore = retriever_cache[file_hash]
        else:
            split_docs = split_text(docs)
            vectorstore = Chroma.from_documents(split_docs, embeddings)
            retriever_cache[file_hash] = vectorstore

        if is_count_question(query):
            return run_tag_pipeline(query, vectorstore)

        retrieved_docs = vectorstore.max_marginal_relevance_search(query, k=7, fetch_k=20)
        if is_count_question(query):
            table_docs = [doc for doc in retrieved_docs if "[테이블" in doc.page_content]
            other_docs = [doc for doc in retrieved_docs if "[테이블" not in doc.page_content]
            retrieved_docs = table_docs + other_docs

        context = format_context(retrieved_docs)
        return self.generator(query, context)

# Gradio Chat Interface
generator = Generator()
rag_pipeline = RAGPipeline(generator)

chatbot = gr.ChatInterface(
    fn=lambda msg, hist, file: rag_pipeline(msg, file),
    title="[LLM] PDF Table RAG+TAG 통합 시스템",
    description="PDF 파일을 업로드하고 질문하세요. 표 및 텍스트 데이터 기반으로 답변합니다.",
    additional_inputs=[gr.File(label="📄 PDF 파일", file_types=[".pdf"])]
)

chatbot.launch()
