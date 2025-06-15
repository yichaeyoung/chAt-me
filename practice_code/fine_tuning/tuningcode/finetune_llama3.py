from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
import json

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

# 준비된 데이터셋 예시
json_path = "qa_data_llama3_100.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
dataset = Dataset.from_list(data)

# LoRA 구성
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./llama3-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    save_total_limit=1,
    learning_rate=2e-4,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    report_to="none"
)

# 학습 시작
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="messages"
)

trainer.train()

# 학습 결과 저장
trainer.model.save_pretrained("./llama3-finetuned")
tokenizer.save_pretrained("./llama3-finetuned")
