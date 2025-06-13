# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from huggingface_hub import login

login(token="my_tocken")  # 자신의 Access Token 입력

# ✅ 설정
BASE_MODEL = "Saxo/Linkbricks-Horizon-AI-Korean-llama-3.1-sft-dpo-8B"
OUTPUT_DIR = "./outputs"
DATA_PATH = "./datasets/train.jsonl"

# ✅ 4bit 양자화 설정 (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ✅ Tokenizer & Model 로드
print("📦 모델 불러오는 중...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# ✅ PEFT: LoRA 구성
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# ✅ 데이터셋 로드
print("📄 학습 데이터 로드 중...")
dataset = load_dataset("json", data_files=DATA_PATH)

# ✅ 전처리
def tokenize_function(examples):
    full_prompt = [f"{q}\n답변: {a}" for q, a in zip(examples["prompt"], examples["completion"])]
    return tokenizer(full_prompt, padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset["train"].map(tokenize_function, batched=True)

# ✅ 데이터 수집기
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ✅ 학습 인자 설정
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ✅ 학습 시작
print("🚀 학습 시작!")
trainer.train()

# ✅ 저장
print("💾 어댑터 저장 중...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 저장 완료: {OUTPUT_DIR}")
