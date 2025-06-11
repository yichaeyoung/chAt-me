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
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS

# PDF page에서 텍스트 추출하는 함수
def extract_text_with_ocr(page):
    text = page.extract_text()
    if not text: # 만약 추출할 텍스트가 없다면
        # PDF page를 이미지로 변환
        image = page.to_image()
        # 이미지에서 OCR 재실행하여 텍스트 추출
        text = pytesseract.image_to_string(image)
    return text

# PDF 및 CSV 파일을 열어서 텍스트를 추출하고 벡터 데이터베이스에 저장하는 함수
def load_and_retrieve_docs(file):
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

    # PyMuPDFLoader를 통해 문서 로드
    pdf_loader = PyMuPDFLoader(file)
    documents_pdf = pdf_loader.load()

    # 텍스트 청크 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents_pdf)

    # FAISS 벡터 DB 생성 및 저장
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("./db/faiss_index")

    return vectorstore.as_retriever()

# 리스트 안의 모든 document 객체 내용을 추출해서 string으로 이어붙여 반환
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
def rag_chain(file, question):
    retriever = load_and_retrieve_docs(file)
    if isinstance(retriever, str):
        return retriever

    retrieved_docs = retriever.get_relevant_documents(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"

    response = ollama.chat(model='llama3.2',
                           messages=[
                               {"role": "system",
                                "content": "You are a helpful assistant. Check the pdf content and answer the question."
                                },
                               {"role": "user",
                                "content": formatted_prompt}
                           ])
    summary = response['message']['content']
    save_to_csv(summary)
    return summary

# 요약 텍스트를 CSV 파일에 저장하는 함수
def save_to_csv(summary):
    df = pd.DataFrame({"Summary": [summary]})
    df.to_csv("summary.csv", index=False)

# Gradio interface
iface = gr.Interface(
    fn=rag_chain,
    inputs=["file", "text"],
    outputs="text",
    title="[LLAMA 3.2] RAG 검색 활용 챗봇 시스템",
    description="PDF 및 CSV 파일을 업로드하고 질문을 입력하면 답변을 생성해드립니다.(영어로!)"
)

iface.launch()

# # 저장한 FAISS DB 불러오기기
# new_vectorstore = FAISS.load_local(
#     "faiss_index",
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# # 문서 추가(기존 벡터DB에 새로운 문서를 동적으로 추가)
# # 추가된 문서는 자동으로 임베딩
# new_docs = [
#     Document(page_content="조선시대의 교육 제도는 성균관 중심이었다.", metadata={"source": "추가"}),
#     Document(page_content="한국 전통 사회에서 글을 읽는 능력은 권력의 상징이었다.", metadata={"source": "추가"})
# ]
# # 문서 추가
# vectorstore.add_documents(new_docs)

# # 유사 문서 검색(사용자 질문과 유사한 문서를 벡터공간에서 검색)
# # (예시)
# query = "가장 심각한 사이버 보안 위험이 뭐야?"
# retrieved_docs = vectorstore.similarity_search(query, k=3)
# # 결과 출력
# print("검색 결과:")
# for doc in retrieved_docs:
#     print(doc.page_content)
#     print('-'*200)
    
# # 저장된 모든 문서 확인하기
# all_docs = new_vectorstore.docstore._dict # 내부에 저장된 모든 문서
# for i, (doc_id, doc) in enumerate(all_docs.items()):
#     print(f"[{i+1}] 문서 ID: {doc_id}")
#     print(doc.page_content[:300]) # 300자까지만 보기
#     print("-" * 50)
    
# # 유사도 점수와 함께 검색
# query = "랜섬웨어 대응 방안은?"
# retrieved_docs = vectorstore.similarity_search_with_score(query, k=2)
# # 점수 추출 + 변환
# for doc, score in retrieved_docs:
#     print('유사도 점수 :', score)
#     print('문서 내용 :', doc.page_content)
#     print('-'*200)
    
