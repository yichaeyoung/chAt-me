import pandas as pd
import json

csv_path = "auto_extracted_table.csv"         # 실제 CSV 경로
jsonl_path = "train.jsonl"

# 열에 맞춰 질문 템플릿 지정
question_templates = {
    "홈페이지":[
    "{val}의 공식 홈페이지 알려주세요.",
    "{val} 홈페이지 주소 뭐야?",
    "{val} 사이트 주소 알려줘.",
    "{val} 웹사이트 어딘가요?",
    "{val} 공식 웹사이트 주소 알려줘요.",
    "{val} 공식 사이트 뭐예요?",
    "{val} 사이트 URL이 뭐지?",
    "{val} 홈피 주소 좀 알려줄래?",
    "{val} 공식 홈페이지 링크 부탁해요.",
    "{val} 웹페이지 주소 알려줘.",
    "{val}의 웹사이트 링크는?",
    "{val} 사이트 주소는?",
    "{val} 홈 주소 좀 알려줘.",
    "{val}의 URL 알려주세요.",
    "{val} 홈페이지 뭐죠?",
    "{val} 웹사이트 링크 있니?",
    "{val} 관련 홈페이지 있어?",
    "{val} 들어갈 수 있는 홈페이지 있어?",
    "{val} 관련 웹사이트 어디야?",
    "{val} 홈페이지 좀 보여줘",
    "{val} 홈페이지 정보 줄 수 있어?",
    "{val} 사이트 url 있나요?",
    "{val} 홈피 주소 알려줘",
    "{val} 홈페이지 찾고 있어요",
    "{val} 홈페이지 링크가 뭐에요?"
    ],     
    "전문분야": [
    "{val}의 전문 분야는 무엇인가요?",
    "{val}의 전문 분야는?",
    "{val}은 어떤 분야를 전문으로 하나요?",
    "{val}의 주요 업무 분야가 뭐야?",
    "{val}은 어떤 일을 주로 해?",
    "{val}의 특화 분야가 뭔가요?",
    "{val}은 어떤 일을 주로 해?",
    ],
    "회사설립일": [
    "{val}의 설립일은 언제인가요?",
    "{val}은 언제 설립되었나요?",
    "{val} 창립일 알려줘",
    "{val}은 몇 년도에 만들어졌어?",
    "{val} 설립된 연도 알려줄래?",
    "{val} 창업일자 알고 싶어",
    ],
    "지역": [
    "{val}은 어느 지역에 있나요?",
    "{val}은 어디에 위치해 있나요?",
    "{val}의 위치 지역은?",
    "{val} 지역이 어디야?",
    "{val} 위치 좀 알려줘",
    "{val} 어느 도시에 있어?",
    ],
    "주소":[
    "{val}의 주소는 어디인가요?",
    "{val} 주소가 어떻게 되나요?",
    "{val}의 정확한 주소는?",
    "{val}의 위치 주소 알려줘",
    "{val} 어디에 있어요?",
    "{val} 건물 주소 뭐야?",
    ],
    "대표전화": [
    "{val}의 전화번호는?",
    "{val}의 고객센터 번호는?",
    "{val} 연락처가 뭐야?",
    "{val} 대표 전화번호 알려줘",
    "{val} 전화번호 좀 줄래?",
    "{val}의 고객센터 번호는?",
    "{val} 연락처가 뭐야?",
    "{val}에 전화하려면 어디로 해야 돼?",
    ],
    "대표자": [
    "{val}의 대표자는 누구인가요?",
    "{val}의 CEO는 누구인가요?",
    "{val} 대표 누구야?",
    "{val}의 책임자는 누구야?",
    "{val} 회장님 성함이?",
    "{val}을 운영하는 사람은 누구인가요?",
    "{val}의 CEO는 누구인가요?",
    ],
    "담당부서": [
    "{val}의 담당 부서는 어디인가요?",
    "{val}에서 이 일 담당하는 부서가 어디인가요?",
    "{val} 담당 부서 알려줘",
    "{val}의 관련 부서가 궁금해요",
    "{val}에 이거 문의하려면 어느 부서야?",
    "{val} 부서 정보 알려줘",
    "{val}에서 이 일 담당하는 부서가 어디인가요?",
    ],
    "AI솔루션": [
    "{val}에서 제공하는 AI 솔루션은?",
    "{val}에서 만드는 AI 제품은 뭐가 있어요?",
    "{val}의 인공지능 서비스는?",
    "{val}에서 하는 AI 관련 사업 알려줘",
    "{val}에서 제공하는 인공지능 기능은?",
    "{val}의 AI 기술은 어떤 게 있나요?",
    "{val}에서 만드는 AI 제품은 뭐가 있어요?",
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
