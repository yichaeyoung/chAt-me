import ollama
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets


# ollama에게 프롬프트를 제공하고 답변을 반환하는 함수
def chat_with_llama(prompt):
    try:
        response = ollama.chat(model='llama3.2', messages=[
            {
                'role':'user',
                'content':prompt,
            },
        ])
        return response['message']['content']
    except Exception as e:
        return f"오류발생:{str(e)}"

# csv를 읽어오는 함수
def load_csv(file_path):
    try:
        # pandas로 csv 파일을 읽음
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return f"CSV 파일 로드 중 오류 발생:{str(e)}"

# 기본 데이터 정보 분석
def analyze_csv(df):
    analysis=""
    # data.shape = 행과 열을 튜플로 반환
    analysis += f"dataset size:{df.shape[0]}raws, {df.shape[1]}columns\n\n"
    analysis += "columns list:\n"+"\n".join(df.columns)+"\n\n"
    analysis += "data type:\n" + df.dtypes.to_string()+"\n\n"
    # 통계량 요약 메서드
    analysis += "descriptive statistics:\n" + df.describe().to_string() + "\n\n"
    
    # missing_data = df.isnull().sum()
    # analysis += "missing_data:\n" + missing_data.to_string() + "\n\n"
    return analysis

# 히스토그램을 그리는 함수
def plot_histogram(df, column):
    plt.figure(figsize=(10, 6))
    df[column].hist()
    plt.title(f'{column} 히스토그램')
    plt.xlabel(column)
    plt.ylabel('빈도')
    plt.show()

# CSV 분석 워크플로우 함수
def csv_analysis_workflow(file_path):
    print("CSV 파일 분석을 시작합니다...")
    # CSV 파일을 로드합니다
    df= load_csv(file_path)
    if isinstance(df, str): # 오류 발생 시
        print(df)
        return

    # 기본 분석을 수행합니다
    analysis_result = analyze_csv(df)
    print(analysis_result)
    
    # Llama에게 분석을 요청합니다
    prompt = f"""Below are the basic analysis results of the CSV file.
    The cold forging process is a type of plastic processing, a forging process that causes plastic deformation at room temperature.
    This is a dataset that contains the manufacturing AI analysis process for quality assurance of cold forging equipment.
    It collects and preprocesses current and inverter alarms, and transfer displacement data from the cold forging equipment, including motors, grippers, cutting lengths, and material transfer.
    Please provide insights based on this data. : \n\n{analysis_result}"""
    
    llama_insights = chat_with_llama(prompt)
    print("Llama Answer:")
    print(llama_insights)

    # 대화형 인터페이스를 시작합니다
    continue_chat_with_llama(file_path, df, analysis_result)


    # Llama와의 대화를 계속하는 함수
def continue_chat_with_llama(file_path, df, analysis_result):
    print("Llama와의 대화를 시작합니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.")

    while True:
        user_input = input("질문을 입력하세요(가급적 영어로): ")

        if user_input.lower() in['quit', 'exit']:
            print("대화를 종료합니다.")
            break
        
        if user_input.startswith("히스토그램:"):
            column = user_input.split(":")[1].strip()
            if column in df.columns:
                plot_histogram(df, column)
            else:
                print(f"'{column}'컬럼을 찾을 수 없습니다.")
        else:
            prompt =f"""다음은 CSV 파일에 대한 질문입니다. 분석 결과를 바탕으로 답변해주세요:
            질문: {user_input}
            분석 결과:
            {analysis_result}"""


            response = chat_with_llama(prompt)
            print("Llama:", response)


# 메인 실행 부분
if __name__ == "__main__":
    file_path = 'D:\\test_data.csv'
    # CSV 파일 경로를 지정합니다
    csv_analysis_workflow(file_path) # CSV 분석 워크플로우를 실행합니다
