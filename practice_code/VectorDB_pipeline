from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.vectorstores import FAISS

# PDF 파일 로드
pdf_path = "2025년 사이버 위협 전망 보고서.pdf"
pdf_loader = PyMuPDFLoader(pdf_path)
# 문서 로드 실행
documents_pdf = pdf_loader.load()
# 출력 확인
print(f"총 {len(documents_pdf)} 개의 페이지가 로드됨")

# CSV 파일 로드
csv_path = "sample.csv"
loader = CSVLoader(file_path=csv_path)
# 문서 로드 실행
documents = loader.load()
print(documents[0].page_content)

# 텍스트 청크 분할
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=500,
chunk_overlap=100,
separators=["\n\n", "\n",". "]
)
split_docs = text_splitter.split_documents(documents_pdf)

# 벡터 DB 저장
vectorstore = FAISS.from_documents(split_docs, embedding_model)
vectorstore.save_local("faiss_index")

# 저장한 FAISS DB 불러오기기
new_vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# 문서 추가(기존 벡터DB에 새로운 문서를 동적으로 추가)
# 추가된 문서는 자동으로 임베딩
new_docs = [
    Document(page_content="조선시대의 교육 제도는 성균관 중심이었다.", metadata={"source": "추가"}),
    Document(page_content="한국 전통 사회에서 글을 읽는 능력은 권력의 상징이었다.", metadata={"source": "추가"})
]
# 문서 추가
vectorstore.add_documents(new_docs)

# 유사 문서 검색(사용자 질문과 유사한 문서를 벡터공간에서 검색)
# (예시)
query = "가장 심각한 사이버 보안 위험이 뭐야?"
retrieved_docs = vectorstore.similarity_search(query, k=3)
# 결과 출력
print("검색 결과:")
for doc in retrieved_docs:
    print(doc.page_content)
    print('-'*200)
    
# 저장된 모든 문서 확인하기
all_docs = new_vectorstore.docstore._dict # 내부에 저장된 모든 문서
for i, (doc_id, doc) in enumerate(all_docs.items()):
    print(f"[{i+1}] 문서 ID: {doc_id}")
    print(doc.page_content[:300]) # 300자까지만 보기
    print("-" * 50)
    
# 유사도 점수와 함께 검색
query = "랜섬웨어 대응 방안은?"
retrieved_docs = vectorstore.similarity_search_with_score(query, k=2)
# 점수 추출 + 변환
for doc, score in retrieved_docs:
    print('유사도 점수 :', score)
    print('문서 내용 :', doc.page_content)
    print('-'*200)
