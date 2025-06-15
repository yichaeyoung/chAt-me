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
from langchain_community.document_loaders import PyMuPDFLoader
import camelot
import ollama
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig

PEFT_MODEL_PATH = "/root/chAtme/chAt-me/practice_code/model/outputs_llama_3_1_coddyddld_train_sample/checkpoint-990"
BASE_MODEL = "Saxo/Linkbricks-Horizon-AI-Korean-llama-3.1-sft-dpo-8B"

retriever_cache = {}

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def get_file_hash(file):
    try:
        file_path = file.name if hasattr(file, "name") else file
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"[ERROR] 해시 생성 실패: {e}")
        return None

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

def format_context(retrieved_docs, max_length=2500):
    context = ""
    for doc in retrieved_docs:
        doc_text = doc.page_content.strip()
        if len(context) + len(doc_text) > max_length:
            break
        context += doc_text + "\n\n"
    return context.strip()

def is_count_question(query):
    patterns = [
        r'개수', r'몇\s*개', r'몇\s*명', r'총\s*개', r'총\s*수', 
        r'총\s*인원', r'수는', r'개는', r'몇\s*곳', r'몇\s*회', 
        r'얼마나', r'많은', r'있는', r'있습니까'
    ]
    return any(re.search(pattern, query) for pattern in patterns)

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

        retrieved_docs = vectorstore.max_marginal_relevance_search(query, k=7, fetch_k=20)
        if is_count_question(query):
            table_docs = [doc for doc in retrieved_docs if "[테이블" in doc.page_content]
            other_docs = [doc for doc in retrieved_docs if "[테이블" not in doc.page_content]
            retrieved_docs = table_docs + other_docs

        context = format_context(retrieved_docs)
        return self.generator(query, context)

class HFGenerator:
    def __init__(self, base_model_path, peft_model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        print(f"🔧 모델 로딩 중... (device: {self.device})")

        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.model = PeftModel.from_pretrained(base_model, peft_model_path) if peft_model_path else base_model
        self.model.eval()

    def __call__(self, query, context):
        prompt = f"""당신은 정확한 문서 분석 AI 비서입니다.
당신의 임무는 아래 문서 내용(context)에 근거해서만 질문에 답하는 것입니다.
만약 문서에 정보가 없거나 애매하다면 "문서에 해당 정보가 없습니다"라고 답하세요.

질문: {query}

컨텍스트:
{context}

명확하고 간결하게 한국어로 답하세요.
"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.2,
                top_p=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ⚙️ 프롬프트 제거하고 실제 답변만 추출
        answer_start = output_text.find("질문:")
        if answer_start != -1:
            output_text = output_text[answer_start:]
            answer_split = output_text.split("명확하고 간결하게 한국어로 답하세요.")
            if len(answer_split) > 1:
                output_text = answer_split[-1].strip()

        return output_text

generator = HFGenerator(BASE_MODEL, PEFT_MODEL_PATH)
rag_pipeline = RAGPipeline(generator)

def chat_callback(msg, hist, file):
    response = rag_pipeline(msg, file)
    return response  # ChatInterface가 history 자동 관리

chatbot = gr.ChatInterface(
    fn=chat_callback,
    title="[LLM] PDF Table RAG+TAG 통합 시스템",
    description="PDF 파일을 업로드하고 질문하세요. 표 및 텍스트 데이터 기반으로 답변합니다.",
    additional_inputs=[gr.File(label="📄 PDF 파일", file_types=[".pdf"])]
)

if __name__ == "__main__":
    chatbot.launch()
