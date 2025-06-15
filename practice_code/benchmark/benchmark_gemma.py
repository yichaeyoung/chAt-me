import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from TAG_RAG import RAGPipeline, HFGenerator  # 기존 RAG 파이프라인

# === 설정 ===
PDF_FILE_PATH = "/root/chAtme/chAt-me/practice_code/benchmark/2025_AI.pdf"
EVAL_FILE_PATH = "test(1).jsonl"
PEFT_MODEL_PATH = "/root/chAtme/chAt-me/practice_code/model/outputs"
BASE_MODEL = "google/gemma-3-4b-it"


# === 벤치마크 실행 ===
def run_benchmark():
    with open(EVAL_FILE_PATH, "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f]

    dummy_file = type("File", (object,), {"name": PDF_FILE_PATH})()
    generator = HFGenerator(BASE_MODEL, PEFT_MODEL_PATH)
    rag_pipeline = RAGPipeline(generator)

    total = len(eval_data)
    correct = 0
    total_latency = 0

    print(f"\n🏁 [GEMMA 4B + LoRA + 8bit] 벤치마크 시작")
    for item in eval_data:
        q = item["prompt"]
        expected = item["completion"].strip()

        start = time.time()
        output = rag_pipeline(q, dummy_file)
        latency = time.time() - start
        total_latency += latency

        is_correct = expected in output
        if is_correct:
            correct += 1

        print(f"\n[Q] {q}\n[A] {output.strip()}\n✔️ 정답 포함 여부: {is_correct}")

    acc = (correct / total) * 100
    avg_time = total_latency / total
    print(f"\n📊 [GEMMA + PEFT + 8bit] 결과 요약:")
    print(f"✅ 정확도: {correct}/{total} ({acc:.2f}%)")
    print(f"⏱️ 평균 응답 시간: {avg_time:.2f}초")


if __name__ == "__main__":
    run_benchmark()
