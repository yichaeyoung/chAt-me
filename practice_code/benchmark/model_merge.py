from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL_PATH = "Saxo/Linkbricks-Horizon-AI-Korean-llama-3.1-sft-dpo-8B"
PEFT_MODEL_PATH = "/root/chAtme/chAt-me/practice_code/model/llama3.1_2epoch_outputs/checkpoint-28044"
MERGED_MODEL_PATH = "/root/chAtme/chAt-me/practice_code/model/llama3.1_merged"

print("📦 base 모델 로딩 중...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map="auto")
model = PeftModel.from_pretrained(base, PEFT_MODEL_PATH)

print("🔗 병합 중...")
merged = model.merge_and_unload()

print(f"💾 병합된 모델 저장: {MERGED_MODEL_PATH}")
os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
merged.save_pretrained(MERGED_MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)
