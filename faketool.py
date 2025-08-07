import streamlit as st
from transformers import pipeline
from collections import Counter
import pandas as pd
import re
import smtplib
from email.mime.text import MIMEText

# Load Zero-Shot Classifier (more reliable output parsing)
from transformers import pipeline



classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

result = classifier(
    "This is a fake news article about elections.",
    candidate_labels=["real", "fake"]
)

print(result)





# Session State
if "log" not in st.session_state:
    st.session_state.log = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "ratings" not in st.session_state:
    st.session_state.ratings = []

# Helper Functions
def extract_keywords(text):
    words = re.findall(r"\b\w{4,}\b", text.lower())
    return words

def send_email_alert(message):
    try:
        sender = "your_email@gmail.com"
        password = "your_app_password"
        receiver = "admin_email@gmail.com"
        msg = MIMEText(message)
        msg["Subject"] = "High Confidence Fake News Alert"
        msg["From"] = sender
        msg["To"] = receiver
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
    except Exception as e:
        print("Email alert failed:", e)

# UI Title
st.set_page_config(page_title="Fake News Classifier", layout="centered")
st.title("🧠 TruthLens AI: Fake News Classifier")
st.write("Check if a headline or article is potentially real or fake using AI.")

# Input
text_input = st.text_area("Enter Headline or Article:", height=150)
image = st.file_uploader("Optional: Upload Screenshot", type=["png", "jpg", "jpeg"])

# Prediction
if st.button("Check Now"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        labels = ["real news", "fake news"]
        result = classifier(text_input, candidate_labels=labels)
        predicted_label = result["labels"][0]
        predicted_score = dict(zip(result["labels"], result["scores"]))[predicted_label] * 100

        st.subheader("🧾 Result")
        st.write(f"**Prediction:** {predicted_label.upper()} ({predicted_score:.2f}% confidence)")

        if predicted_label == "fake news" and predicted_score > 90:
            send_email_alert(f"⚠️ Fake News Detected!\n\nText: {text_input}\nConfidence: {predicted_score:.2f}%")

        st.session_state.log.append({"text": text_input, "label": predicted_label, "confidence": round(predicted_score, 2)})

# Sidebar Trending
st.sidebar.header("🔥 Trending Keywords")
if st.session_state.log:
    all_text = " ".join([entry["text"] for entry in st.session_state.log])
    keywords = extract_keywords(all_text)
    for word, freq in Counter(keywords).most_common(10):
        st.sidebar.write(f"{word} ({freq})")

# Feedback
st.write("---")
st.subheader("🗳 Feedback")
feedback = st.text_input("Any suggestions?")
if st.button("Submit Feedback"):
    st.session_state.feedback.append(feedback)
    st.success("Thank you for your feedback!")

# Rating
st.subheader("🌟 Rate this Tool")
rating = st.slider("Your Rating (1 = Poor, 5 = Excellent)", 1, 5)
if st.button("Submit Rating"):
    st.session_state.ratings.append(rating)
    st.success("Thanks for rating us!")

# Stats
st.write("---")
st.subheader("📊 Tool Usage Stats")
st.write(f"**Total Checks:** {len(st.session_state.log)}")
if st.session_state.ratings:
    st.write(f"**Average Rating:** {sum(st.session_state.ratings)/len(st.session_state.ratings):.2f}")
else:
    st.write("No ratings yet.")

# Download Logs
if st.button("Download Logs"):
    df = pd.DataFrame(st.session_state.log)
    df.to_csv("log.csv", index=False)
    st.success("Logs downloaded as log.csv")



