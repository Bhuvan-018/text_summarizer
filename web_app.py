import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


st.set_page_config(page_title="AI Text Summarizer", layout="wide")
st.title("AI Text Summarizer")
st.caption("Paste a long article and generate a concise abstractive summary.")


@st.cache_resource
def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model


def summarize_text(
    text: str,
    tokenizer,
    model,
    max_input_length: int,
    max_output_length: int,
    min_output_length: int,
    num_beams: int,
):
    prompt = text.strip()
    if tokenizer.name_or_path.startswith("google-t5/") or "t5" in tokenizer.name_or_path.lower():
        prompt = f"summarize: {prompt}"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_input_length, truncation=True)
    output_ids = model.generate(
        **inputs,
        max_length=max_output_length,
        min_length=min_output_length,
        num_beams=num_beams,
        length_penalty=2.0,
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


with st.sidebar:
    st.header("Model Settings")
    model_path = st.text_input("Model path", value="./finetuned_summarizer")
    max_input_length = st.slider("Max input length", min_value=128, max_value=1024, value=512, step=64)
    max_output_length = st.slider("Max output length", min_value=32, max_value=256, value=128, step=8)
    min_output_length = st.slider("Min output length", min_value=8, max_value=128, value=30, step=2)
    num_beams = st.slider("Beam size", min_value=1, max_value=8, value=4, step=1)

user_input = st.text_area("Paste your article here", height=320)
if st.button("Summarize", type="primary"):
    if not user_input.strip():
        st.warning("Please paste some text.")
    else:
        with st.spinner("Loading model and generating summary..."):
            tokenizer, model = load_model(model_path)
            summary = summarize_text(
                text=user_input,
                tokenizer=tokenizer,
                model=model,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
                min_output_length=min_output_length,
                num_beams=num_beams,
            )
        st.subheader("Summary")
        st.write(summary)
