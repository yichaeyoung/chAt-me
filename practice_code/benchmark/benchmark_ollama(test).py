import json
import time
from TAG_RAG import RAGPipeline, Generator # import 해올 python 파일, 함수

PDF_FILE_PATH = "/2025_AI.pdf" # pdf 이름

generator = Generator()
rag_pipeline = RAGPipeline(generator)

def run_benchmark():
    with open("test_convert.jsonl", "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f]

    total = len(eval_data)
    correct = 0
    total_latency = 0

    dummy_file = type("File", (object,), {"name": PDF_FILE_PATH})()

    for item in eval_data:
        q = item["question"]
        expected = item["answer"].strip()

        start = time.time()
        output = rag_pipeline(q, dummy_file)
        latency = time.time() - start
        total_latency += latency

        is_correct = expected in output
        if is_correct:
            correct += 1

        print(f"\n[Q] {q}\n[A] {output.strip()}\n✔️ 정답 포함 여부: {is_correct}")

    print("\n📊 벤치마크 결과:")
    print(f"✅ 정확도: {correct}/{total} ({(correct/total)*100:.2f}%)")
    print(f"⏱️ 평균 응답 시간: {total_latency/total:.2f}초")

if __name__ == "__main__":
    run_benchmark()
