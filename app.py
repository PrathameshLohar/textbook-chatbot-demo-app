import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

st.set_page_config(page_title="LLaMA 2 Textbook Chatbot")

@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Prathamesh25/Llama-2-7b-textbook-chatbot",
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("Prathamesh25/Llama-2-7b-textbook-chatbot")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

model, tokenizer = load_model()

st.title("📘 LLaMA 2 Textbook QA Bot")

user_input = st.text_area("Ask a question from your textbook:", height=100)

if st.button("Get Answer") and user_input:
    prompt = f"<s>[INST] {user_input} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.markdown("### Answer:")
    st.write(response.split("[/INST]")[-1].strip())
