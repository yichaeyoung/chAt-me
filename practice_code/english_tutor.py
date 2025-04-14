from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Ollama 모델 초기화
model = OllamaLLM(model="llama3.2")

# 프롬프트 템플릿 정의
template = """
You are an English vocabulary tutor. 
When given a word, explain its meaning in simple terms, and provide an example sentence. 
If the word has multiple meanings, explain each with examples. 
Here's the input:

Word: {word}

Output format:
1. Definition: [Brief explanation in simple English]
2. Example Sentence: [A sentence using the word in context]"""

# LangChain 체인 구성
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def get_definition(word):
    return chain.invoke({"word": word})

if __name__ == '__main__':
    while True:
        word = input('enter word: ')
        if word == '/bye':
            break
        print(get_definition(word))
        print()
