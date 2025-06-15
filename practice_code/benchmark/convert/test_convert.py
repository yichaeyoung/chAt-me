import pandas as pd
import json

csv_path = "auto_extracted_table.csv"         # 실제 CSV 경로
jsonl_path = "test(1).jsonl"

# 열에 맞춰 질문 템플릿 지정
question_templates = {
    "홈페이지": [
        "{val} 관련 웹 주소를 알려주세요.",
    ],
    "전문분야": [
        "{val}이 다루는 기술 분야는 무엇인가요?",
    ],
    "회사설립일": [
        "{val}의 설립 연도는 몇 년인가요?",
    ],
    "지역": [
        "{val}의 기반 지역은 어디인가요?",
    ],
    "주소": [
        "{val}이 위치한 장소는 어디인가요?",
    ],
    "대표전화": [
        "{val}에 연락하려면 어느 번호로 해야 하나요?",
    ],
    "대표자": [
        "{val} 대표의 성함은 무엇인가요?",
    ],
    "담당부서": [
        "{val}의 해당 담당 부서를 알고 싶어요.",
    ],
    "AI솔루션": [
        "{val}의 주요 AI 기능들을 소개해주세요."
    ]
}

df = pd.read_csv(csv_path)

with open(jsonl_path, "w", encoding="utf-8") as f:
    for i, row in df.iterrows():
        company_name = str(row["기업명"]).strip()
        for col in question_templates:
            val = str(row.get(col, "")).strip()
            if val and val != "nan":
                for template in question_templates[col]:
                    question = template.replace("{val}", company_name)
                    answer = val
                    f.write(json.dumps({"prompt": question, "completion": " " + answer}, ensure_ascii=False) + "\n")

print("✅ JSON 변환 완료:", jsonl_path)
