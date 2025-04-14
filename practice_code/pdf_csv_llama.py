# ==========================
# 필수 라이브러리 임포트
import gradio as gr                      # 웹 UI 프레임워크
import pandas as pd                     # CSV 파일 처리
from langchain.docstore.document import Document  # LangChain용 문서 객체
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 긴 문서를 나누는 유틸
from langchain_community.vectorstores import Chroma  # 벡터 DB 저장소 (in-memory or local)
from langchain_ollama import OllamaEmbeddings        # Ollama 모델용 임베딩 처리
import ollama                            # LLM API (로컬에서 llama3.2 모델 사용)
import pytesseract                       # OCR 라이브러리 (PDF 이미지 처리용)
from pdf2image import convert_from_path  # PDF → 이미지로 변환
import hashlib                          # 파일 해시 처리 (캐싱에 사용)
import os                               # 파일 시스템 관련 작업

# ==========================
# 시스템 설정
POPPLER_PATH = r"C:\Program Files\poppler-xx\poppler-24.08.0\Library\bin"
# → Windows에서는 pdf2image가 PDF를 이미지로 변환할 때 poppler 경로 필요

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# → OCR 처리를 위한 Tesseract 엔진 경로 지정

# ==========================
# 캐시 (재처리 방지용)
retriever_cache = {}
# → 파일 해시를 키로 retriever를 저장, 재사용 가능


# ==========================
# 파일 해시 생성
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
# → 파일 내용을 기반으로 고유한 해시값 생성 (retriever 캐시 키로 사용)


# ==========================
# PDF → 텍스트 (OCR 방식)
def extract_text_from_pdf_with_ocr(file_path):
    text = ""
    try:
        images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        for i, image in enumerate(images):
            print(f"Processing page {i+1}")
            page_text = pytesseract.image_to_string(image)
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        return f"Error processing PDF with OCR: {e}"

    if not text.strip():
        return "No text could be extracted from the PDF using OCR."
    return text
# → PDF 파일을 페이지별로 이미지로 바꾸고 OCR 처리하여 텍스트 추출


# ==========================
# CSV → 텍스트
def extract_text_from_csv(file_path):
    encodings_to_try = ["utf-8", "cp949", "euc-kr"]
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            return df.to_string(index=False)  # 표 전체를 문자열로 반환
        except Exception as e:
            print(f" Failed with encoding {enc}: {e}")
    return "Error: Unable to read CSV file with common encodings (utf-8, cp949, euc-kr)."
# → CSV 파일의 인코딩을 자동으로 시도하며 읽고 문자열로 반환

# ==========================
# 문서 처리 및 임베딩/검색 준비
def load_and_retrieve_docs(file):
    file_path = file.name if hasattr(file, 'name') else file
    file_hash = get_file_hash(file_path)

    # 캐시 확인
    if file_hash in retriever_cache:
        print("📁 캐시된 retriever 사용 중...")
        return retriever_cache[file_hash]

    # 파일 종류에 따라 텍스트 추출 방식 선택
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf_with_ocr(file_path)
    elif file_path.endswith(".csv"):
        text = extract_text_from_csv(file_path)
    else:
        return "Unsupported file type. Please upload a PDF or CSV."

    # 오류 또는 빈 텍스트 처리
    if isinstance(text, str) and text.startswith("Error"):
        return text
    if not text.strip():
        return "No text found in the file."

    # LangChain 문서 생성 및 분할
    docs = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 임베딩 생성 및 벡터 저장소 생성
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    retriever = vectorstore.as_retriever()
    retriever_cache[file_hash] = retriever  # 캐시에 저장

    print("✅ 새 retriever 생성 완료")
    return retriever


# ==========================
# 문서 포맷팅
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# → 관련 문서 리스트를 텍스트로 변환


# ==========================
# 결과 세이브파일 한 파일에 누적 저장
# def save_to_csv(question, result, file_name="result.csv"):
#     df_new = pd.DataFrame([{"Question": question, "Answer": result}])
#     if os.path.exists(file_name):
#         df_existing = pd.read_csv(file_name)
#         df_combined = pd.concat([df_existing, df_new], ignore_index=True)
#         df_combined.to_csv(file_name, index=False)
#     else:
#         df_new.to_csv(file_name, index=False)

# 결과 CSV파일 여러개 만들어서 저장
def save_to_csv(question, result, base_file_name="result.csv"):
    df_new = pd.DataFrame([{"Question": question, "Answer": result}])

    # 기존 파일이 없으면 새로 저장
    if not os.path.exists(base_file_name):
        df_new.to_csv(base_file_name, index=False)
        print(f"Saved to {base_file_name}")
        return

    # 기존 파일이 비어있으면 거기에 저장
    df_existing = pd.read_csv(base_file_name)
    if df_existing.empty:
        df_new.to_csv(base_file_name, index=False)
        print(f"Saved to {base_file_name}")
        return

    # 결과 중복 방지: result1.csv, result2.csv 등으로 저장
    i = 1
    while True:
        new_file_name = f"result{i}.csv"
        if not os.path.exists(new_file_name):
            df_new.to_csv(new_file_name, index=False)
            print(f"Saved to {new_file_name}")
            break
        else:
            df_check = pd.read_csv(new_file_name)
            if df_check.empty:
                df_new.to_csv(new_file_name, index=False)
                print(f"Saved to {new_file_name}")
                break
        i += 1


# ==========================
# RAG 기반 질문 응답 체인
def rag_chain(message, history, file):
    retriever = load_and_retrieve_docs(file)
    if isinstance(retriever, str):  # 에러 메시지인 경우
        return retriever

    # 관련 문서 검색
    retrieved_docs = retriever.get_relevant_documents(message)
    formatted_context = format_docs(retrieved_docs)

    # 프롬프트 생성
    formatted_prompt = f"Question: {message}\n\nContext: {formatted_context}"

    # Ollama LLM 호출
    response = ollama.chat(model='llama3.2',
                           messages=[
                               {"role": "system",
                                "content": "You are a helpful assistant. Check the content and answer the question."},
                               {"role": "user",
                                "content": formatted_prompt}
                           ])
    result = response['message']['content']
    save_to_csv(message, result)
    return result


# ==========================
# Gradio UI 구성 및 실행
chatbot = gr.ChatInterface(
    fn=rag_chain,  # 메시지를 처리하는 함수
    title="[LLAMA 3.2] RAG 검색 활용 챗봇 시스템",
    description="PDF 또는 CSV 파일을 업로드하고 질문을 입력하세요. 파일은 캐시에 저장되어 빠른 재질문이 가능합니다.",
    additional_inputs=[gr.File(label="PDF 또는 CSV 파일", file_types=[".pdf", ".csv"])],
)

chatbot.launch()  # 웹서버 실행
