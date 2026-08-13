import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

# app.py is in the project root
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide"
)

# ============================================================
# LOAD FILES
# ============================================================

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

# ============================================================
# HELPER FUNCTIONS
# ============================================================

suspicious_words = [
    "free",
    "win",
    "winner",
    "cash",
    "prize",
    "urgent",
    "claim",
    "click",
    "call",
    "now",
    "reward",
    "selected",
    "account",
    "verify",
    "limited",
    "offer",
    "congratulations"
]


def find_suspicious_words(message):
    found = []

    message_lower = message.lower()

    for word in suspicious_words:
        if word in message_lower:
            found.append(word)

    return found


def get_spam_probability(message):
    try:
        probability = model.predict_proba([message])[0][1] * 100
        return probability
    except Exception:
        return None


def display_risk(probability):
    if probability is None:
        st.info("Confidence score unavailable for this saved model.")
        return

    st.metric(
        "Spam Probability",
        f"{probability:.2f}%"
    )

    st.progress(
        min(100, max(0, int(probability)))
    )

    if probability < 30:
        st.success("🟢 Low Risk")

    elif probability < 70:
        st.warning("🟡 Medium Risk")

    else:
        st.error("🔴 High Risk")


# ============================================================
# TITLE
# ============================================================

st.title("📧 Email Spam Detection with Explainable NLP")

st.markdown("""
This proof-of-concept system uses Natural Language Processing to classify
messages as **legitimate** or **spam / suspicious**.

The application also highlights suspicious terms and provides simple
explainability through spam-associated keywords.
""")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📘 Project Menu")

    st.write("""
    ### How to Use

    1. Paste an email or message
    2. Click **Detect Spam**
    3. Review the prediction
    4. Check the spam probability
    5. Review suspicious words
    6. Explore model insights
    """)

    st.info(
        "Cybersecurity NLP proof-of-concept."
    )

    st.warning(
        "This application is not a production phishing detection system."
    )

# ============================================================
# TABS
# ============================================================

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

    if st.button(
        "Detect Spam",
        type="primary"
    ):

        if user_message.strip() == "":

            st.warning(
                "Please enter a message first."
            )

        else:

            prediction = model.predict(
                [user_message]
            )[0]

            probability = get_spam_probability(
                user_message
            )

            suspicious_found = find_suspicious_words(
                user_message
            )

            st.subheader("📋 Detection Result")

            # ------------------------------------------------
            # Main prediction
            # ------------------------------------------------

            if prediction == 1:

                st.error(
                    "### Spam / Suspicious Message Detected 🚨"
                )

            else:

                st.success(
                    "### Legitimate Message ✅"
                )

            # ------------------------------------------------
            # Probability / risk
            # ------------------------------------------------

            display_risk(probability)

            # ------------------------------------------------
            # Recommended action
            # ------------------------------------------------

            st.write("### Recommended Action")

            if prediction == 1:

                st.warning("""
                - Do not click unexpected links
                - Do not download unknown attachments
                - Verify the sender independently
                - Avoid providing personal or financial information
                - Report or delete the message if appropriate
                """)

            else:

                st.info("""
                The message appears legitimate according to the model.

                You should still verify unexpected links, attachments
                and sender information before taking action.
                """)

            # ------------------------------------------------
            # Suspicious words
            # ------------------------------------------------

            st.write(
                "### Suspicious Words Detected"
            )

            if suspicious_found:

                st.warning(
                    ", ".join(suspicious_found)
                )

            else:

                st.success(
                    "No predefined suspicious keywords detected."
                )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Suspicious Keyword Count",
                    len(suspicious_found)
                )

            with col2:

                st.metric(
                    "Message Length",
                    len(user_message)
                )

            # ------------------------------------------------
            # Explanation
            # ------------------------------------------------

            st.write(
                "### Why was this prediction made?"
            )

            st.write("""
            The machine learning model converts the message into
            numerical TF-IDF features.

            Logistic Regression then uses learned word weights to
            determine whether the language resembles patterns commonly
            found in spam messages.

            The suspicious keywords above provide an additional simple
            explanation for the user.
            """)

            # ------------------------------------------------
            # Input message
            # ------------------------------------------------

            with st.expander(
                "View analysed message"
            ):

                st.write(user_message)

# ============================================================
# TAB 2 — SAMPLE MESSAGES
# ============================================================

with tab2:

    st.subheader("📊 Sample Messages")

    st.write("""
    Select a message from the held-out test dataset and run it
    through the trained model.
    """)

    sample_df = pd.DataFrame({
        "Message": sample_messages
    })

    st.dataframe(
        sample_df.head(100),
        use_container_width=True
    )

    selected_index = st.selectbox(
        "Choose a sample message:",
        sample_df.index
    )

    selected_message = sample_df.loc[
        selected_index,
        "Message"
    ]

    st.write("### Selected Message")

    st.write(selected_message)

    if st.button(
        "Classify Selected Message"
    ):

        prediction = model.predict(
            [selected_message]
        )[0]

        probability = get_spam_probability(
            selected_message
        )

        st.subheader(
            "📋 Sample Classification"
        )

        if prediction == 1:

            st.error(
                "### Spam / Suspicious Message Detected 🚨"
            )

        else:

            st.success(
                "### Legitimate Message ✅"
            )

        display_risk(probability)

        suspicious_found = find_suspicious_words(
            selected_message
        )

        if suspicious_found:

            st.write(
                "### Suspicious Words"
            )

            st.warning(
                ", ".join(suspicious_found)
            )

# ============================================================
# TAB 3 — MODEL INSIGHTS
# ============================================================

with tab3:

    st.subheader("🔎 Model Insights")

    # --------------------------------------------------------
    # Keywords
    # --------------------------------------------------------

    st.write(
        "### Top Spam-Associated Keywords"
    )

    top_keywords = keyword_df.head(15)

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.barh(
        top_keywords["Keyword"],
        top_keywords["Spam_Weight"]
    )

    ax.invert_yaxis()

    ax.set_title(
        "Top Spam-Associated Keywords"
    )

    ax.set_xlabel(
        "Logistic Regression Weight"
    )

    st.pyplot(fig)

    plt.close(fig)

    st.caption("""
    Larger positive weights indicate terms that contribute more strongly
    towards the model predicting spam.
    """)

    # --------------------------------------------------------
    # Keyword table
    # --------------------------------------------------------

    st.write(
        "### Keyword Importance Table"
    )

    st.dataframe(
        keyword_df.head(30),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Optional saved confidence data
    # --------------------------------------------------------

    if confidence_df is not None:

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Test Messages Analysed",
                len(confidence_df)
            )

        with col2:

            if "SpamProbability" in confidence_df.columns:

                avg_probability = (
                    confidence_df[
                        "SpamProbability"
                    ].mean()
                    * 100
                )

                st.metric(
                    "Average Spam Probability",
                    f"{avg_probability:.2f}%"
                )

        st.write(
            "### Highest Risk Messages"
        )

        if (
            "SpamProbability"
            in confidence_df.columns
        ):

            highest_risk = (
                confidence_df
                .sort_values(
                    "SpamProbability",
                    ascending=False
                )
                .head(20)
            )

            st.dataframe(
                highest_risk,
                use_container_width=True
            )

        if (
            "Actual" in confidence_df.columns
            and
            "Predicted" in confidence_df.columns
        ):

            st.write(
                "### Misclassified Messages"
            )

            wrong_predictions = confidence_df[
                confidence_df["Actual"]
                !=
                confidence_df["Predicted"]
            ]

            st.dataframe(
                wrong_predictions.head(20),
                use_container_width=True
            )

    else:

        st.info("""
        Optional confidence analysis data was not found.
        The main prediction system will still work normally.
        """)

# ============================================================
# TAB 4 — ABOUT
# ============================================================

with tab4:

    st.subheader(
        "ℹ️ About This Project"
    )

    st.markdown("""
    ### Project Overview

    This proof-of-concept project explores spam detection using
    Natural Language Processing and machine learning.

    ### Machine Learning Task

    - Binary text classification
    - Legitimate vs spam messages
    - TF-IDF text vectorisation
    - Logistic Regression classification

    ### Explainability

    Logistic Regression coefficients are used to identify terms
    associated with spam predictions.

    A separate suspicious-keyword mechanism is also included to
    provide users with an easier-to-understand explanation.

    ### Cybersecurity Relevance

    Spam and suspicious messages are common attack vectors for
    scams, social engineering and phishing.

    Machine learning can help filter suspicious content before
    users interact with potentially harmful messages.

    ### Limitations

    - The dataset mainly contains SMS-style messages
    - It may not represent modern business email
    - The model does not inspect URLs
    - It does not inspect attachments
    - It does not analyse email headers
    - It does not assess sender reputation
    - It does not detect domain spoofing
    - Keyword matching is simple and rule-based
    - This is not a production email security system
    """)

    st.divider()

    st.subheader(
        "🎣 Phishing Detection Limitations"
    )

    st.write("""
    This application identifies spam-like **text patterns**.

    It does not perform full phishing detection because it does not
    analyse technical indicators such as:

    - URL reputation
    - Domain age
    - Sender authentication
    - SPF / DKIM / DMARC
    - Email headers
    - Attachments
    - Redirect chains
    - Website content
    """)

    st.warning("""
    This application is for educational and defensive cybersecurity
    purposes only.

    It should not be used as a production spam, phishing,
    or email-security system.
    """)