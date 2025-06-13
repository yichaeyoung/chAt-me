import pandas as pd
import json

csv_path = "auto_extracted_table.csv"         # 실제 CSV 경로
jsonl_path = "test.jsonl"

# 열에 맞춰 질문 템플릿 지정
question_templates = {
    "홈페이지": [
        "{val} 관련 웹 주소를 알려주세요.",
        "{val} 기업의 온라인 페이지는 어디인가요?",
        "{val}의 웹사이트는 어떻게 접속하나요?",
        "{val}에 대한 공식 URL을 알려주세요.",
        "{val} 인터넷 홈페이지를 찾고 있어요.",
        "{val}의 웹 포털 주소를 알려줄 수 있나요?",
        "{val}은 어떤 웹사이트를 운영하나요?",
        "{val} 방문 가능한 사이트가 궁금해요."
    ],
    "전문분야": [
        "{val}이 다루는 기술 분야는 무엇인가요?",
        "{val}은 어떤 산업에 집중하고 있나요?",
        "{val}의 핵심 사업 영역은 무엇인가요?",
        "{val}의 비즈니스 전문 분야가 궁금해요.",
        "{val}의 주요 역량이 뭔가요?",
        "{val}은 어떤 분야에서 활동하나요?"
    ],
    "회사설립일": [
        "{val}은 어느 해에 설립되었나요?",
        "{val}의 설립 연도는 몇 년인가요?",
        "{val}이 창업된 시기는 언제인가요?",
        "{val} 설립 시점을 알고 싶어요.",
        "{val}의 창립된 연도는?",
        "{val}은 어떤 시기에 시작되었나요?"
    ],
    "지역": [
        "{val}의 기반 지역은 어디인가요?",
        "{val}은 주로 어느 지역에 있나요?",
        "{val} 본사가 위치한 도시는?",
        "{val}의 활동 거점은 어디인가요?",
        "{val}의 지역 정보가 궁금합니다.",
        "{val}이 있는 지역명을 알려주세요."
    ],
    "주소": [
        "{val}의 구체적인 위치를 알려주세요.",
        "{val} 사무실 주소는 어디인가요?",
        "{val}이 위치한 장소는 어디인가요?",
        "{val}의 실제 주소 정보가 궁금해요.",
        "{val} 회사의 주소를 알려주세요.",
        "{val}이 위치한 정확한 장소는?"
    ],
    "대표전화": [
        "{val}에 연락하려면 어느 번호로 해야 하나요?",
        "{val}의 공식 전화번호가 궁금해요.",
        "{val}에 문의할 전화번호가 필요해요.",
        "{val} 고객 문의 번호가 뭐죠?",
        "{val}과 통화하려면 어디로 전화하나요?",
        "{val} 상담 번호를 알고 싶어요."
    ],
    "대표자": [
        "{val}을 이끄는 사람은 누구인가요?",
        "{val} 대표의 성함은 무엇인가요?",
        "{val}에서 경영을 맡은 분은 누구죠?",
        "{val}을 운영 중인 인물은 누구인가요?",
        "{val} 회사의 수장은 누구인가요?",
        "{val}의 CEO는 어떤 분인가요?"
    ],
    "담당부서": [
        "{val}에서 이 일을 맡은 부서는 어디인가요?",
        "{val} 관련 부서 담당 정보 알려주세요.",
        "{val}에서는 어느 부서가 이 역할을 합니까?",
        "{val}의 해당 담당 부서를 알고 싶어요.",
        "{val}의 전담 부서는 어떤 곳인가요?",
        "{val}에선 누가 이 부문을 맡나요?"
    ],
    "AI솔루션": [
        "{val}이 제공하는 인공지능 기능은 어떤 게 있나요?",
        "{val}의 AI 제품군이 궁금합니다.",
        "{val}에서 개발한 AI 기술이 있나요?",
        "{val}이 운영하는 AI 서비스는 어떤 게 있나요?",
        "{val}에서 제공하는 지능형 솔루션은 무엇인가요?",
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
