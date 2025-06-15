import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr
import os

# ==== 설정 ====
BASE_MODEL = "google/gemma-3-4b-it"
PEFT_MODEL_PATH = "./outputs_model2"
USE_PEFT = True
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HUGGINGFACE_TOKEN"] = "my_tocken"

# ==== Tokenizer & Model 로딩 ====
print("📦 Tokenizer 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 8bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# 모델 로딩
print("📦 모델 로딩 중...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": device.index if device.type == "cuda" else "cpu"},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

if USE_PEFT:
    print("🪄 PEFT 모델 로딩 중...")
    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)
else:
    model = base_model
model.eval()

# ==== 채팅 함수 ====
def chat_and_qa(user_input, pdf_file, csv_file, history):
    try:
        file_context = ""
        if pdf_file is not None:
            file_context += "[PDF 파일 분석 완료]\n"
        if csv_file is not None:
            file_context += "[CSV 파일 분석 완료]\n"

        prompt = (
            "<bos>"
            "<start_of_turn>user\n"
            + file_context
            + f"{user_input}\n<end_of_turn>\n<start_of_turn>model\n"
        )

        print("\n📥 [디버깅] 입력 프롬프트:\n", prompt)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
                streamer=streamer,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 출력에서 직접 디코딩해 반환 (streamer는 출력만 하고 결과 문자열은 안 줌)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reply = output_text.split("<start_of_turn>model")[-1].strip()
        history.append([user_input, reply])
        return history, history

    except Exception as e:
        err_msg = f"❌ 오류 발생: {str(e)}"
        print(err_msg)
        history.append([user_input, err_msg])
        return history, history

# ==== Gradio UI ====
with gr.Blocks() as demo:
    gr.Markdown("## 💬 GEMMA-3-4b-it + LoRA 대화 인터페이스")
    chatbox = gr.Chatbot()
    user_input = gr.Textbox(placeholder="질문을 입력해주세요", label="질문")
    pdf_file = gr.File(label="PDF 업로드 (선택)", file_types=[".pdf"])
    csv_file = gr.File(label="CSV 업로드 (선택)", file_types=[".csv"])
    history = gr.State([])
    send_btn = gr.Button("전송")

    send_btn.click(
        fn=chat_and_qa,
        inputs=[user_input, pdf_file, csv_file, history],
        outputs=[chatbox, history]
    )

# ==== 서버 실행 ====
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
