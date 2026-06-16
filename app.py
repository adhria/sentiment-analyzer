import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Page config
st.set_page_config(
    page_title="Multilingual Sentiment Analyzer",
    page_icon="🌍",
    layout="centered"
)

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

st.title("🌍 Multilingual Sentiment Analyzer")
st.write("Type any text in **any language** and get instant sentiment analysis.")

with st.spinner("Loading model (first time takes ~30 seconds)..."):
    tokenizer, model = load_model()

# Input
text = st.text_area("Enter text here:", height=150, placeholder="Type in English, Spanish, Arabic, Hindi, French...")

if st.button("Analyze", use_container_width=True):
    if text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            scores = F.softmax(outputs.logits, dim=1)[0]
            labels = ["Negative", "Neutral", "Positive"]
            scores_dict = {labels[i]: float(scores[i]) for i in range(len(labels))}

            top_label = max(scores_dict, key=scores_dict.get)
            top_score = scores_dict[top_label]

            # Display result
            emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}[top_label]
            color = {"Positive": "green", "Neutral": "orange", "Negative": "red"}[top_label]

            st.markdown(f"### Result: :{color}[{emoji} {top_label}]")
            st.metric("Confidence", f"{top_score * 100:.1f}%")

            st.write("#### All Scores:")
            for label, score in scores_dict.items():
                st.progress(score, text=f"{label}: {score * 100:.1f}%")