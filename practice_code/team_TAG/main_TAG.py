import gradio as gr
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama
import pdfplumber
import pytesseract
#from PIL import Image # 이미지 처리 라이브러리
import hashlib
import os  # 수정 가능해보임

# 캐시 저장소
retriever_cache = {}

# 파일 해시 생성
def get_file_hash(file):
    file_path = file.name if hasattr(file, "name") else file
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# OCR로 PDF 페이지에서 텍스트 추출
def extract_text_with_ocr(page):
    text = page.extract_text()
    if not text:
        image = page.to_image(resolution=300).original
        text = pytesseract.image_to_string(image)
    return text

# PDF 테이블 추출 후 CSV 저장
def extract_tables_from_pdf(file, save_csv_path="auto_extracted_table.csv"):
    try:
        with pdfplumber.open(file) as pdf:
            all_tables = []
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        df = pd.DataFrame(table)
                        all_tables.append(df)
        if all_tables:
            combined_df = pd.concat(all_tables, ignore_index=True)
            combined_df.to_csv(save_csv_path, index=False, header=False)
            print(f"PDF 테이블이 CSV로 저장되었습니다: {save_csv_path}")
            return combined_df
        else:
            print("테이블 없음.")
            return None
    except Exception as e:
        print(f"PDF 테이블 추출 실패: {e}")
        return None

# 파일 로딩 및 벡터 인덱스 생성
def load_and_retrieve_docs(file):
    file_hash = get_file_hash(file)
    if file_hash in retriever_cache:
        print("캐시된 retriever 사용 중...")
        return retriever_cache[file_hash]

    ext = os.path.splitext(file.name)[1].lower()
    text = ""
    docs = []

    try:
        if ext == ".pdf":
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = extract_text_with_ocr(page)
                    if page_text:
                        text += page_text
            docs = [Document(page_content=text)]

            # 자동 테이블 추출 및 CSV 저장
            extract_tables_from_pdf(file)

        elif ext == ".csv":
            df = pd.read_csv(file)
            docs = [Document(page_content=str(row)) for _, row in df.iterrows()]
        else:
            return "지원되지 않는 파일 형식입니다."
    except Exception as e:
        return f"파일 읽기 오류: {e}"

    if not docs:
        return "문서에서 텍스트를 추출하지 못했습니다."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    retriever_cache[file_hash] = retriever
    return retriever

# 문서를 텍스트로 변환
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 질의 유형 분류
def classify_query_type(message):
    response = ollama.chat(
        # model="llama3.2", 한국어버전 쓰기위해 수정
        model="benedict/linkbricks-llama3.1-korean:8b",
        messages=[
            {"role": "system", "content": "다음 사용자의 질문이 어떤 유형인지 판별하세요. 선택지는: Text2SQL, RAG, TAG. 유형만 한 단어로 답변하세요."},
            {"role": "user", "content": message}
        ]
    )
    return response['message']['content'].strip().lower()

# TAG 파이프라인
def run_tag_pipeline(message, retriever):
    relevant_docs = retriever.get_relevant_documents(message)
    formatted_table = format_docs(relevant_docs)
    prompt = f"""다음 테이블 데이터를 참고하여 질문에 답변하세요.

    질문: {message}

    테이블:
    {formatted_table}
    """
    response = ollama.chat(
        # model="llama3.2",
        model="benedict/linkbricks-llama3.1-korean:8b",
        messages=[
            {"role": "system", "content": "당신은 테이블 기반 정보를 바탕으로 추론하는 LLM입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']

# RAG 파이프라인
def run_rag_pipeline(message, retriever):
    retrieved_docs = retriever.get_relevant_documents(message)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {message}\n\nContext: {formatted_context}"
    response = ollama.chat(
        # model="llama3.2",
        model="benedict/linkbricks-llama3.1-korean:8b",
        messages=[
            {"role": "system", "content": "PDF 또는 문서 내용을 참고하여 질문에 답하세요."},
            {"role": "user", "content": formatted_prompt}
        ]
    )
    return response['message']['content']

# 전체 통합 처리
def combined_chain(message, history, file_list):
    file = file_list[0] if file_list else None
    if not file:
        return "파일을 업로드해야 합니다."

    retriever = load_and_retrieve_docs(file)
    if isinstance(retriever, str):
        return retriever

    query_type = classify_query_type(message)
    print(f"질의 유형 분류 결과: {query_type}")

    if "tag" in query_type:
        answer = run_tag_pipeline(message, retriever)
    else:
        answer = run_rag_pipeline(message, retriever)

    save_to_csv(answer)
    return answer

# 답변 결과 저장
def save_to_csv(summary):
    df = pd.DataFrame({"Summary": [summary]})
    df.to_csv("summary.csv", index=False)

# Gradio 앱
chatbot = gr.ChatInterface(
    fn=combined_chain,
    title="[LLAMA 3.2] RAG + TAG 통합 챗봇 시스템",
    description="PDF 또는 테이블 파일(CSV)을 업로드하고 질문을 입력하세요. 적절한 방식(TAG 또는 RAG)으로 자동 처리됩니다.",
    additional_inputs=[gr.File(label="PDF 또는 CSV 파일", file_types=[".pdf", ".csv"], file_count="multiple")]
)

chatbot.launch()
