import json
import time
import requests

OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"

def query_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]

def run_benchmark():
    with open("eval_set.jsonl", "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f]

    total = len(eval_data)
    correct = 0
    total_latency = 0

    for item in eval_data:
        q = item["question"]
        expected = item["answer"].strip()

        start = time.time()
        output = query_ollama(q)
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
