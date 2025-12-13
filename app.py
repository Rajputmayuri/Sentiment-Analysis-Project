import nltk
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

# Load model and vectorizer
model = joblib.load("sentiment_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

# Stopwords (keep negations)
STOPWORDS = set(stopwords.words("english"))
for w in ["not", "no", "nor"]:
    STOPWORDS.discard(w)


# Contraction expansion
def expand_contractions(text):
    contractions = {
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "can not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "shouldn't": "should not",
        "couldn't": "could not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
    }
    for c, e in contractions.items():
        text = re.sub(rf"\b{c}\b", e, text, flags=re.IGNORECASE)
    return text


# Cleaning function
def clean_text(s):
    s = s.lower()
    s = expand_contractions(s)
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [w for w in s.split() if w not in STOPWORDS]
    return " ".join(tokens)


# UI
st.title("🎬 Sentiment Analysis App")
st.write("Enter a movie review and predict its sentiment.")

review = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][prediction]

        if prediction == 1:
            st.success(f"😊 Positive Sentiment (Confidence: {probability:.2f})")
        else:
            st.error(f"😞 Negative Sentiment (Confidence: {probability:.2f})")
