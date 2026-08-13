import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide"
)

# -----------------------------
# Load files
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "spam_model.pkl")

@st.cache_data
def load_sample_messages():
    return joblib.load(MODELS_DIR / "sample_messages.pkl")

@st.cache_data
def load_spam_keywords():
    return joblib.load(MODELS_DIR / "spam_keywords.pkl")

@st.cache_data
def load_confidence_data():
    path = MODELS_DIR / "spam_confidence_data.pkl"
    if path.exists():
        return joblib.load(path)
    return None

model = load_model()
sample_messages = load_sample_messages()
keyword_df = load_spam_keywords()
confidence_df = load_confidence_data()

# -----------------------------
# Helper function
# -----------------------------
suspicious_words = [
    "free", "win", "winner", "cash", "prize",
    "urgent", "claim", "click", "call", "now",
    "reward", "selected", "account", "verify",
    "limited", "offer", "congratulations"
]

def find_suspicious_words(message):
    found = []
    message_lower = message.lower()

    for word in suspicious_words:
        if word in message_lower:
            found.append(word)

    return found

# -----------------------------
# Title
# -----------------------------
st.title("📧 Email Spam Detection with Explainable NLP")

st.markdown("""
This proof-of-concept system uses Natural Language Processing to classify messages as
**legitimate** or **spam / suspicious**.

It also highlights suspicious words and shows interpretable spam-associated keywords.
""")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("📘 Project Menu")

    st.write("""
    ### How to Use
    1. Paste an email or message
    2. Click Detect Spam
    3. Review the prediction and confidence
    4. Check suspicious words and model insights
    """)

    st.info("Cybersecurity NLP proof-of-concept.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📧 Spam Detection",
    "📊 Sample Messages",
    "🔎 Model Insights",
    "ℹ️ About"
])

# ============================================================
# TAB 1 — SPAM DETECTION
# ============================================================
with tab1:
    st.subheader("📧 Paste Message or Email Text")

    example_messages = [
        "",
        "Congratulations! You have won a free cash prize. Click now to claim.",
        "Hi, are we still meeting tomorrow afternoon?",
        "URGENT! Your account has been selected for a reward. Verify now.",
        "Can you send me the project report when you get a chance?"
    ]

    selected_example = st.selectbox(
        "Try an example message:",
        example_messages
    )

    user_message = st.text_area(
        "Enter message text:",
        value=selected_example,
        height=180
    )

    if st.button("Detect Spam"):
        if user_message.strip() == "":
            st.warning("Please enter a message first.")
        else:
            prediction = model.predict([user_message])[0]

            try:
                probability = model.predict_proba([user_message])[0][1] * 100
            except Exception:
                probability = None

            st.subheader("📋 Detection Result")

            if prediction == 1:
                st.error("### Spam / Suspicious Message Detected 🚨")
            else:
                st.success("### Legitimate Message ✅")

            if probability is not None:
                st.metric("Spam Probability", f"{probability:.2f}%")
                st.progress(int(probability))

                if probability < 30:
                    st.success("🟢 Low Risk")
                elif probability < 70:
                    st.warning("🟡 Medium Risk")
                else:
                    st.error("🔴 High Risk")

            else:
                st.info("Confidence score unavailable for this saved model.")

            suspicious_found = find_suspicious_words(user_message)

            st.write("### Suspicious Words Detected")

            if suspicious_found:
                st.warning(", ".join(suspicious_found))

                st.metric(
                    "Suspicious Keyword Count",
                    len(suspicious_found)
                )
            else:
                st.success("No predefined suspicious keywords detected.")

                st.metric(
                    "Suspicious Keyword Count",
                    0
                )

            st.write("### Input Message")
            st.write(user_message)

            st.write("### Why was this prediction made?")

            st.write("""
            The machine learning model converts the message into numerical TF-IDF features.
            Words that frequently appear in spam messages contribute more strongly towards a spam prediction.
            The suspicious keywords shown above provide a simple explanation of why the message may have been classified as spam.
            """)

            st.metric(
                "Message Length",
                len(user_message)
            )

# ============================================================
# TAB 2 — SAMPLE MESSAGES
# ============================================================
with tab2:
    st.subheader("📊 Sample Messages")

    sample_df = pd.DataFrame({
        "Message": sample_messages
    })

    st.dataframe(sample_df.head(100))

    selected_index = st.selectbox(
        "Choose a sample message:",
        sample_df.index
    )

    selected_message = sample_df.loc[selected_index, "Message"]

    st.write("### Selected Message")
    st.write(selected_message)

    prediction = model.predict([selected_message])[0]

    try:
        probability = model.predict_proba([selected_message])[0][1] * 100
    except Exception:
        probability = None

    if prediction == 1:
        st.error("### Spam / Suspicious Message Detected 🚨")
    else:
        st.success("### Legitimate Message ✅")

    if prediction == 1:
        st.warning("""
        **Recommended Action**

        • Do not click any links.

        • Do not download attachments.

        • Verify the sender before responding.

        • Delete or report the message if suspicious.
        """)
    else:
        st.info("""
            **Recommended Action**

            The message appears legitimate, but always verify unexpected links,
            attachments and sender information.
            """)

    if probability is not None:
        st.info(f"Spam Probability: {probability:.2f}%")
    else:
        st.info("Confidence score unavailable for this saved model.")

# ============================================================
# TAB 3 — MODEL INSIGHTS
# ============================================================
with tab3:
    st.subheader("🔎 Model Insights")

    st.write("### Top Spam-Associated Keywords")

    top_keywords = keyword_df.head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        top_keywords["Keyword"],
        top_keywords["Spam_Weight"]
    )
    ax.invert_yaxis()
    ax.set_title("Top Spam-Associated Keywords")
    ax.set_xlabel("Spam Weight")

    st.pyplot(fig)

    st.write("### Keyword Table")
    st.dataframe(keyword_df.head(30))

    if confidence_df is not None:
        st.write("### Highest Risk Messages")

        st.dataframe(
            confidence_df.sort_values(
                "SpamProbability",
                ascending=False
            ).head(20)
        )

        st.write("### Misclassified Messages")

        wrong_predictions = confidence_df[
            confidence_df["Actual"] != confidence_df["Predicted"]
        ]

        st.dataframe(wrong_predictions.head(20))

# ============================================================
# TAB 4 — ABOUT
# ============================================================
with tab4:
    st.subheader("ℹ️ About This Project")

    st.write("""
    This proof-of-concept project explores spam detection using Natural Language Processing.

    **Machine Learning Task**
    - Binary text classification
    - Classifies messages as legitimate or spam
    - Uses TF-IDF vectorisation and Logistic Regression

    **Explainability**
    - Spam-associated keywords are extracted from the model coefficients
    - Suspicious words are highlighted for user understanding

    **Cybersecurity Relevance**
    Spam and suspicious messages are common attack vectors for scams and phishing attempts.
    Machine learning can help identify risky text patterns.

    **Limitations**
    - Dataset is SMS-style and may not fully represent modern phishing emails
    - The model does not inspect links, attachments, sender reputation or email headers
    - Suspicious keyword matching is simple and rule-based
    - This is not a production email security tool
    """)

    st.write("""
    ### Phishing Detection

    This application detects spam-like language patterns only.

    It does **not** inspect:

    - URLs
    - Attachments
    - Email headers
    - Sender reputation
    - Domain spoofing

    Therefore, sophisticated phishing emails may still evade detection.
    """)

    st.warning("""
    This app is for educational and defensive cybersecurity purposes only.
    It should not be used as a production spam or phishing detection system.
    """)