from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama
import pdfplumber
import pytesseract
import pandas as pd
import hashlib
import os
import shutil
import uuid

#uvicorn test:app --host 0.0.0.0 --port 8000 --reload
#위 코드 터미널에 실행하면 실행됨됨

app = FastAPI()

# CORS 설정: OpenWebUI 또는 프론트엔드 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever_cache = {}

def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def extract_text_with_ocr(page):
    text = page.extract_text()
    if not text:
        image = page.to_image(resolution=300).original
        text = pytesseract.image_to_string(image)
    return text

def extract_tables_from_pdf(file_path, save_csv_path="auto_extracted_table.csv"):
    try:
        with pdfplumber.open(file_path) as pdf:
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

def load_and_retrieve_docs(file_path):
    file_hash = get_file_hash(file_path)
    if file_hash in retriever_cache:
        print("캐시된 retriever 사용 중...")
        return retriever_cache[file_hash]

    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    docs = []

    try:
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = extract_text_with_ocr(page)
                    if page_text:
                        text += page_text
            docs = [Document(page_content=text)]
            extract_tables_from_pdf(file_path)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def classify_query_type(message):
    response = ollama.chat(
        model="benedict/linkbricks-llama3.1-korean:8b",
        messages=[
            {"role": "system", "content": "다음 사용자의 질문이 어떤 유형인지 판별하세요. 선택지는: Text2SQL, RAG, TAG. 유형만 한 단어로 답변하세요."},
            {"role": "user", "content": message}
        ]
    )
    return response['message']['content'].strip().lower()

def run_tag_pipeline(message, retriever):
    relevant_docs = retriever.get_relevant_documents(message)
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

def run_rag_pipeline(message, retriever):
    retrieved_docs = retriever.get_relevant_documents(message)
    formatted_context = format_docs(retrieved_docs)
    prompt = f"Question: {message}\n\nContext: {formatted_context}"
    response = ollama.chat(
        model="benedict/linkbricks-llama3.1-korean:8b",
        messages=[
            {"role": "system", "content": "PDF 또는 문서 내용을 참고하여 질문에 답하세요."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']

def save_to_csv(summary):
    df = pd.DataFrame({"Summary": [summary]})
    df.to_csv("summary.csv", index=False)

@app.post("/chat")
async def chat(message: str = Form(...), file: UploadFile = File(...)):
    try:
        # 고유 파일명으로 저장
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        temp_path = f"./uploads/{file_id}{file_ext}"
        os.makedirs("./uploads", exist_ok=True)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        retriever = load_and_retrieve_docs(temp_path)
        if isinstance(retriever, str):
            return JSONResponse(content={"error": retriever}, status_code=400)

        query_type = classify_query_type(message)
        print(f"질의 유형: {query_type}")

        if "tag" in query_type:
            answer = run_tag_pipeline(message, retriever)
        else:
            answer = run_rag_pipeline(message, retriever)

        save_to_csv(answer)
        return {"answer": answer}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: dict):
    try:
        message = request["messages"][-1]["content"]

        # 이전에 업로드한 파일이 이미 처리되어 있다고 가정
        file_path = "./uploads/last_used_file.pdf"  # 또는 CSV

        retriever = load_and_retrieve_docs(file_path)
        if isinstance(retriever, str):
            return {"error": retriever}

        query_type = classify_query_type(message)
        if "tag" in query_type:
            answer = run_tag_pipeline(message, retriever)
        else:
            answer = run_rag_pipeline(message, retriever)

        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ],
            "model": "benedict/linkbricks-llama3.1-korean:8b"
        }
    except Exception as e:
        return {"error": str(e)}
