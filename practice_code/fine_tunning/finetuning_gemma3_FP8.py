from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset


from huggingface_hub import login
login(token="my_token") 

# ✅ 설정
BASE_MODEL = "google/gemma-3-4b-it"
OUTPUT_DIR = "./outputs"
DATA_PATH = "./train.jsonl"

# ✅ 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # fork 이후 tokenizers 경고 방지

# ✅ 양자화 설정 (8bit)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# ✅ Accelerator 초기화
accelerator = Accelerator()

# ✅ Tokenizer 로딩
print("📦 Tokenizer 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ✅ 모델 로딩
print("📦 모델 로딩 중...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# ✅ PEFT: LoRA 적용
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

# ✅ 데이터셋 로딩
print("📄 학습 데이터 로딩 중...")
dataset = load_dataset("json", data_files=DATA_PATH)

def tokenize_function(examples):
    full_prompt = [f"{q}\n답변: {a}" for q, a in zip(examples["prompt"], examples["completion"])]
    return tokenizer(full_prompt, padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset["train"].map(tokenize_function, batched=True)

# ✅ 데이터 수집기
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ 학습 인자
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    ddp_find_unused_parameters=True,
)

# ✅ Trainer 구성
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

# ✅ 결과 저장
print("💾 모델 저장 중...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 저장 완료: {OUTPUT_DIR}")
