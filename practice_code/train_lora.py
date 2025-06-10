# practice_code/train_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import os

# 사전 학습된 모델 지정
model_id = "meta-llama/Llama-2-7b-hf"

# 모델 및 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# LoRA 구성
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 학습 데이터 로딩
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="./data/train.txt",
    block_size=512
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./lora_llama",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# 학습 시작
trainer.train()

# 모델 저장
model.save_pretrained("./lora_llama")
tokenizer.save_pretrained("./lora_llama")
