import os
import re
import pandas as pd
import streamlit as st
from collections import Counter
from transformers import pipeline
import smtplib
from email.mime.text import MIMEText

# Lazy load model: Initially None
classifier = None

def get_classifier():
    """Load the AI model only when needed."""
    global classifier
    if classifier is None:
        with st.spinner("🔄 Loading AI Model... Please wait..."):
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier

# Session State Initialization
if "log" not in st.session_state:
    st.session_state.log = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "ratings" not in st.session_state:
    st.session_state.ratings = []

# Predefined Trending Topics
predefined_keywords = [
    "elections", "government", "health", "technology",
    "AI", "economy", "COVID", "sports", "education", "climate"
]

# Helper Functions
def extract_keywords(text):
    return re.findall(r"\b\w{4,}\b", text.lower())

def send_email_alert(message):
    try:
        sender = os.environ.get("ALERT_EMAIL_SENDER", "your_email@gmail.com")
        password = os.environ.get("ALERT_EMAIL_PASSWORD", "your_app_password")
        receiver = os.environ.get("ALERT_EMAIL_RECEIVER", "admin_email@gmail.com")

        msg = MIMEText(message)
        msg["Subject"] = "⚠️ High Confidence Fake News Alert"
        msg["From"] = sender
        msg["To"] = receiver

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
    except Exception as e:
        st.error(f"Email alert failed: {e}")

# UI Setup
st.set_page_config(page_title="TruthLens AI", layout="centered", page_icon="🧠")
st.title("And You Believed That? 🤦‍♀️: Fake News Classifier")
st.write("📰 **Fact-check your headline or article instantly using AI.**")

# Input
text_input = st.text_area("✍️ Enter Headline or Article:", height=150)
image = st.file_uploader("🖼️ Optional: Upload Screenshot", type=["png", "jpg", "jpeg"])

# Prediction
if st.button("🔍 Check Now"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        clf = get_classifier()  # Load model only when needed
        with st.spinner("🧠 Analyzing with AI..."):
            labels = ["real news", "fake news"]
            result = clf(text_input, candidate_labels=labels)
            predicted_label = result["labels"][0]
            predicted_score = dict(zip(result["labels"], result["scores"]))[predicted_label] * 100

            st.subheader("🧾 Result")
            st.success(f"🔎 **Prediction:** `{predicted_label.upper()}`  \n📊 **Confidence:** `{predicted_score:.2f}%`")

            if predicted_label == "fake news" and predicted_score > 90:
                send_email_alert(
                    f"⚠️ Fake News Detected!\n\nText: {text_input}\nConfidence: {predicted_score:.2f}%"
                )

            st.session_state.log.append({
                "text": text_input,
                "label": predicted_label,
                "confidence": round(predicted_score, 2)
            })

# Sidebar: Trending Keywords
st.sidebar.header("🔥 Trending Keywords")
all_text = " ".join([entry["text"] for entry in st.session_state.log])
keywords = extract_keywords(all_text) + predefined_keywords
keyword_freq = Counter(keywords)
for word, freq in keyword_freq.most_common(10):
    st.sidebar.write(f"🔹 {word.capitalize()} ({freq})")

# Show Past Predictions
if st.session_state.log:
    st.write("---")
    st.subheader("📁 Past Predictions")
    for item in reversed(st.session_state.log[-5:]):  
        st.markdown(f"📰 **Text:** {item['text'][:100]}...")
        st.markdown(f"🧠 **Prediction:** `{item['label']}` | 💯 **Confidence:** `{item['confidence']}%`")
        st.markdown("---")

# Feedback Section
st.subheader("🗳️ Your Feedback Matters")
feedback = st.text_input("💬 Any suggestions?")
if st.button("✅ Submit Feedback"):
    st.session_state.feedback.append(feedback)
    st.success("🙌 Thank you for your valuable feedback!")

# Rating Section
st.subheader("🌟 Rate this Tool")
rating = st.slider("📈 Your Rating (1 = Poor, 5 = Excellent)", 1, 5)
if st.button("🎯 Submit Rating"):
    st.session_state.ratings.append(rating)
    st.success("🎉 Thanks for rating us!")

# Stats Display
st.write("---")
st.subheader("📊 Usage Insights")
st.write(f"🧮 **Total Checks:** {len(st.session_state.log)}")
if st.session_state.ratings:
    avg_rating = sum(st.session_state.ratings) / len(st.session_state.ratings)
    st.write(f"⭐ **Average Rating:** {avg_rating:.2f}")
else:
    st.write("⭐ No ratings yet.")

# Download Logs
if st.button("📥 Download Logs"):
    df = pd.DataFrame(st.session_state.log)
    df.to_csv("log.csv", index=False)
    st.success("📁 Logs downloaded as `log.csv`")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by Riya , Shivani , Liesha for Puch AI Hackathon")
