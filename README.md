# chAt&me

# 1. 프로젝트 개요 (개발 목적, 개발 기간 등)

- Open LLM를 활용한 생산공정 데이터 관리 플랫폼을 구축하는 것으로, 생산 제품의 전 생애 주기를 지원하는 멀티모달 AI 플랫폼 구축을 목표로 함
- 기업과 연계한 실증적 프로젝트 수행함으로써 GEN-AI Foundation, Agent, 트윈 환경 구축을 경험하고자 함

# 2. Team Members (팀원 및 팀 소개)
  
| 이채영 | 김나영 | 박수빈 | 원민 |
|:------:|:------:|:------:|:------:|
| <img src="https://github.com/user-attachments/assets/86b2f0a0-4f78-4295-b312-8b93bfe75287" alt="이채영" width="150"> | <img src="https://github.com/user-attachments/assets/d222b45a-2d3a-41e3-b1f5-cc8cea8e61d6" alt="김나영" width="150"> | <img src="https://github.com/user-attachments/assets/7cbaa641-332f-495d-b3fc-a27679eeb173" alt="박수" width="150"> | <img src="https://github.com/user-attachments/assets/409d635f-9ffb-4aee-9330-bf4ab14b43af" alt="원민" width="150"> |
| PL | FE | BE | AI |
| [GitHub](https://github.com/yichaeyoung) | [GitHub](https://github.com/knyjs0710) | [GitHub](https://github.com/ps9b) | [GitHub](https://github.com/wonmin9211) |


# 3. 실행 가이드

- 실행 환경 세팅

```plaintext
pip install ollama chromadb langchain langchain-community tornado gradio pandas pdfplumber pytesseract pillow
```

- 임베딩 모델 설정

```plaintext
ollama pull mxbai-embed-large
```
# 4. 폴더 구조

```plaintext
project/
├── practice_code/
│   ├── pdf_extraction.py    # pdf를 챗봇이 요구사항에 맞게 요약후 csv 파일로 저장 (원민)
│   ├── english_tutor.py     # ollama와 langchain을 이용한 영어 단어장 (박수빈)
│   ├── csv_ollama.py        # ollama에게 csv 파일을 학습시키는 연습 코드 (박수빈)
│   └── pdf_csv_llama.py     # pdf_extraction.py 코드 변화 시켜 챗봇 스타일로 pdf, csv파일을 읽고 질문 대답 (김나영)
├── 주간발표연습/
│   ├── 1주차/               # 1주차 발표 ppt
│   └── 2주차/               # 2주차 발표 ppt
├── 주간보고일지/             # 주간보고일지 파일들
└── README.md                # 프로젝트 개요 및 사용법
```

# 5. 주요 기능

### 5.1 GEN-AI Foundation

### 5.2 GEN-AI Agent

### 5.3 GEN-AI Meta

# 6. 기술 스택 및 개발 환경

<div style="display:flex; flex-direction:row;">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=LangChain&logoColor=white" />
  <img src="https://img.shields.io/badge/visual%20studio%20code-%23007ACC.svg?&style=flat&logo=visual%20studio%20code&logoColor=white" />
  <img src="https://img.shields.io/badge/github-%23181717.svg?&style=flaat&logo=github&logoColor=white" />
  <img src="https://img.shields.io/badge/notion-%23000000.svg?&style=flat&logo=notion&logoColor=white" />
</div><br>

# 7. 아키텍쳐

# 8. 코드 작성 규칙
1. 함수명 : 함수명 길더라도 이름만 보고 알 수 있게 작명

ex) extract_tables_from_pdf

2. 함수 위에 주석 달기

'''
@ 함수 설명 어쩌고저쩌고 기능 설명 작성 (간결하게)
매개 변수 있다면 뭐 받아와야 하는지 작성
이하 세부 설명 작성
'''

3. 주석 많이 많이 달기
