import streamlit as st
import pickle
import re

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


st.title("Movie Review Sentiment Checker")
st.write("Paste a review below and it'll tell you if it reads positive or negative.")

review = st.text_area("Your review", height=150)

if st.button("Check sentiment"):
    if review.strip() == "":
        st.warning("Type something first")
    else:
        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        if pred == 1:
            st.success(f"Positive review ({prob[1]*100:.1f}% confidence)")
        else:
            st.error(f"Negative review ({prob[0]*100:.1f}% confidence)")
