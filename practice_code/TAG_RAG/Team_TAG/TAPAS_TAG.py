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
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from transformers import TapasTokenizer, TapasModel
import camelot
import torch
import ollama
import warnings

# 경고 무시 설정
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`max_length` is ignored when `padding`")

# --- 임베딩 모델 설정 ---
# 텍스트용 SBERT 한국어 특화 모델
text_embeddings = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-Medium-extended-patent2024-hn"
)
# 테이블용 TAPAS 모델
tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wikisql-supervised")
model     = TapasModel.from_pretrained("google/tapas-base-finetuned-wikisql-supervised")

# DataFrame 단위 테이블 임베딩 함수 (TAPAS)
def embed_table(df: pd.DataFrame, query: str) -> list[float]:
    MAX_ROWS = 5
    MAX_COLS = 10
    if df.empty:
        return [0.0] * model.config.hidden_size
    df_trunc = df.head(MAX_ROWS).iloc[:, :MAX_COLS].astype(str)
    inputs = tokenizer(
        table=df_trunc,
        queries=[query],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    outputs = model(**inputs)
    cls_vec = outputs.last_hidden_state[:, 0, :].detach().squeeze(0)
    return cls_vec.tolist()

# LangChain 테이블 임베딩 래퍼
from langchain.embeddings.base import Embeddings
from typing import List

class TableEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = []
        for content in texts:
            try:
                df = pd.read_fwf(io.StringIO(content))
            except Exception:
                df = pd.DataFrame()
            embs.append(embed_table(df, ""))
        return embs

    def embed_query(self, query: str) -> List[float]:
        return embed_table(pd.DataFrame(), query)

# --- PDF 테이블 추출 ---
def extract_tables_from_pdf(file_path: str, save_csv_path: str = "auto_extracted_table.csv") -> list[pd.DataFrame]:
    """
    PDF에서 표를 추출하고, 잘리는 열이 있는지 감지하여 보정한 후 CSV로 저장합니다.
    """
    dfs: list[pd.DataFrame] = []
    # 1) pdfplumber 구조적 테이블
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                for tbl in page.find_tables():
                    data = tbl.extract()
                    if not data or not data[0]:
                        continue
                    cols = [str(x or '').strip() for x in data[0]]
                    rows = []
                    for r in data[1:]:
                        row = [str(cell or '').strip() for cell in r]
                        if any(row):
                            rows.append(row)
                    if rows:
                        dfs.append(pd.DataFrame(rows, columns=cols))
    except Exception as e:
        print(f"[WARN] pdfplumber table extraction failed: {e}")
    # 2) Camelot fallback
    try:
        tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
        if not tables:
            tables = camelot.read_pdf(file_path, pages='all', flavor='stream')
        for table in tables:
            df = table.df.copy()
            df.columns = df.iloc[0].fillna('').astype(str).tolist()
            df = df[1:].reset_index(drop=True)
            dfs.append(df)
    except Exception as e:
        print(f"[WARN] Camelot extraction failed: {e}")
    # 3) 잘린 열 감지 및 보정
    if dfs:
        col_counts = [df.shape[1] for df in dfs]
        max_cols = max(col_counts)
        for i, df in enumerate(dfs):
            if df.shape[1] < max_cols:
                print(f"[WARN] Table {i} has {df.shape[1]} columns, expected {max_cols}. Padding missing columns.")
                # 빈 문자열 열로 보정
                extra_cols = [''] * (max_cols - df.shape[1])
                df = pd.concat([df, pd.DataFrame([extra_cols] * len(df), columns=[f'col_pad_{j}' for j in range(max_cols - df.shape[1])])], axis=1)
                dfs[i] = df
        # CSV 저장 (패딩 후)
        try:
            combined = pd.concat(dfs, ignore_index=True)
            combined.to_csv(save_csv_path, index=False)
            print(f"✅ CSV로 테이블 저장 완료: {save_csv_path}")
        except Exception as e:
            print(f"[WARN] CSV 저장 실패: {e}")
    return dfs

# --- PDF 텍스트 추출 ---
def extract_text_from_pdf(file_path: str) -> list[Document]:
    docs = []
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        if any(len(d.page_content.strip()) < 100 for d in docs):
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    txt = page.extract_text() or ''
                    if len(txt.strip()) < 100:
                        img = page.to_image(resolution=300).original
                        txt = pytesseract.image_to_string(img, lang='kor+eng')
                    docs.append(Document(page_content=f"[페이지 {i+1}]\n" + txt))
    except:
        pass
    return docs

# --- 중복 방지를 위한 파일 해시 ---
def get_file_hash(file) -> str:
    try:
        path = file.name if hasattr(file, 'name') else file
        return hashlib.md5(open(path, 'rb').read()).hexdigest()
    except:
        return None

# --- RAG+TAG 파이프라인 통합 ---
class RAGPipeline:
    def __init__(self, text_embeddings, table_embeddings):
        self.text_embeddings = text_embeddings
        self.table_embeddings = table_embeddings
        self.cache = {}

    def __call__(self, query: str, file) -> str:
        fh = get_file_hash(file)
        # 문서 로드
        text_docs = extract_text_from_pdf(file.name)
        table_dfs = extract_tables_from_pdf(file.name)
        # 캐시 생성
        if fh not in self.cache:
            tvs = Chroma.from_documents(
                text_docs,
                embedding=self.text_embeddings,
                collection_name=f"{fh}_text"
            )
            # 테이블 DataFrame -> Document
            tdocs = [Document(page_content=df.to_string(), metadata={"source":f"table_{i}"})
                     for i, df in enumerate(table_dfs)]
            tavs = Chroma.from_documents(
                tdocs,
                embedding=self.table_embeddings,
                collection_name=f"{fh}_table"
            )
            self.cache[fh] = (tvs, tavs)
        tvs, tavs = self.cache[fh]
        # 검색
        thits = tvs.max_marginal_relevance_search(query, k=7)
        tahits = tavs.similarity_search(query, k=7)
        # 컨텍스트 조합
        ctx_text = "\n\n".join(d.page_content for d in thits)
        ctx_table = "\n\n".join(doc.page_content for doc in tahits)
        context = ctx_text + "\n\n[테이블]\n" + ctx_table
        # 프롬프트
        system_prompt = f"당신은 PDF 문서를 분석하는 전문 AI 비서입니다. 제공된 컨텍스트만을 기반으로 정확히 답변하세요.\n\n컨텍스트:\n{context}"
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":query}
        ]
        res = ollama.chat(model='benedict/linkbricks-llama3.1-korean:8b', messages=messages)
        return res['message']['content']

# --- Gradio UI ---
if __name__ == "__main__":
    pipeline = RAGPipeline(text_embeddings=text_embeddings, table_embeddings=TableEmbeddings())
    iface = gr.ChatInterface(
        fn=lambda msg, hist, file: pipeline(msg, file),
        title="[LLM] PDF RAG+TAG 통합",
        description="PDF 내용(텍스트+테이블)을 통합 검색 및 질의응답하세요.",
        additional_inputs=[gr.File(label="📄 PDF 파일", file_types=[".pdf"])]
    )
    iface.launch()
