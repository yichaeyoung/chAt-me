import nest_asyncio
nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# 1. PDF 파싱 (LlamaParse는 여전히 API 키 필요)
parser = LlamaParse(
    api_key="llx-Isuaxh34cWDEidEacIQOh9vIEfXYnxqhrich2SSpG4jWOpwL",  
    result_type="text",
    verbose=True,
)
documents = parser.load_data("/home/coddyddld/Desktop/workspace/chAtme/dataset/1) 2025년 Ai바우처 공급기업 Pool (1627개사)-수정_250213.pdf")  # ← 여기에 파싱할 PDF 경로

# 2. HuggingFace 임베딩 설정
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. 벡터 인덱스 생성
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# 4. HuggingFace LLM 연결
llm = HuggingFaceLLM(
    model_name="tiiuae/falcon-7b-instruct",  # 또는 mistralai/Mistral-7B-Instruct-v0.1
    tokenizer_name="tiiuae/falcon-7b-instruct",
    device_map="auto",  # CPU만 사용하려면 "cpu"
    #context_window=2048,
    #max_new_tokens=256,
    context_window=1024,
    max_new_tokens=128,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
)

# 5. 질의엔진 생성 및 질의
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("이 문서에서 '서울특별시'에 해당되는 항목은 모두 몇개야?")
print(response)
