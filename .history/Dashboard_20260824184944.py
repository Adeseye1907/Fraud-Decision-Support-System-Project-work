# ============================================================
# INTERPRETABLE FRAUD DETECTION & OPERATIONAL DECISION SUPPORT
# ============================================================
#
# DEVELOPMENT OF ADAPTIVE AND INTERPRETABLE MACHINE LEARNING
# FRAMEWORK FOR REAL-TIME FRAUD DETECTION AND OPERATIONAL
# DECISION SUPPORT IN DIGITAL BANKING
#
# Developed by: Adeseye Samuel Ademola
#
# IMPORTANT:
# This is a prototype/reference implementation.
# Operational actions such as APPROVE, VERIFY, BLOCK and
# ESCALATE are simulated DSS recommendations and are NOT
# connected to a real banking core system.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

from kafka import KafkaConsumer
from kafka.errors import KafkaError


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

SHIELD_ICON = "Shield Check.png"

st.set_page_config(
    page_title="Fraud Detection & Operational DSS",
    page_icon=SHIELD_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. IMAGE FILES
# ============================================================

OVERVIEW_ICON = "dashboard overview.png"
FRAUD_CAUGHT_ICON = "shield check 2.png"
FRAUD_MISSED_ICON = "Alert.png"
FALSE_ALARM_ICON = "bell warning 2.jpg"
VALUE_PROTECTED_ICON = "money shield.png"
VALUE_RISK_ICON = "money warning.jpg"
MODEL_ICON = "target.png"
PRECISION_ICON = "guage 3.jpg"
THRESHOLD_ICON = "slider2.jpg"
TRANSACTION_ICON = "seacrh icon 3.png"
TREND_ICON = "trend chart 2.png"
XAI_ICON = "brain circuit 6.jpg"
LIVE_ICON = "live pulse 2.png"
REFRESH_ICON = "refresh circle.jpg"
INFO_ICON = "info circle.jpg"
AI_ASSISTANT_ICON = "Ai assistant 2.jpg"


# ============================================================
# 4. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 19px;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .welcome-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 20px;
    }

    .status-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.20);
    }

    .risk-high {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(255, 0, 0, 0.08);
        border: 1px solid rgba(255, 0, 0, 0.25);
    }

    .risk-low {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(0, 180, 0, 0.08);
        border: 1px solid rgba(0, 180, 0, 0.25);
    }

    .login-box {
        max-width: 500px;
        margin: auto;
        padding: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. SESSION STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "transactions" not in st.session_state:
    st.session_state.transactions = []

if "action_log" not in st.session_state:
    st.session_state.action_log = []

if "historical_df" not in st.session_state:
    st.session_state.historical_df = None

if "show_forgot_password" not in st.session_state:
    st.session_state.show_forgot_password = False


# ============================================================
# 6. DEMO LOGIN DETAILS
# ============================================================
#
# YOU CAN CHANGE THESE DETAILS.
#
# Current demo account:
#
# Email: admin@frauddss.com
# Password: FraudDSS123
#
# Change the values below if you want.
# ============================================================

DEMO_USERS = {
    "admin@frauddss.com": {
        "password": "FraudDSS123",
        "name": "Fraud DSS Administrator",
        "role": "Fraud Analyst"
    },

    "analyst@frauddss.com": {
        "password": "Analyst123",
        "name": "Fraud Analyst",
        "role": "Fraud Analyst"
    }
}


# ============================================================
# 7. LOGIN FUNCTION
# ============================================================

def login_screen():

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown(
            '<div class="login-box">',
            unsafe_allow_html=True
        )

        st.image(SHIELD_ICON, width=90)

        st.markdown(
            """
            <div class="main-title">
                Fraud DSS
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="subtitle">
                Interpretable Fraud Detection & Operational Decision Support
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        st.subheader("Welcome")

        st.write(
            "Sign in to access the fraud detection and operational "
            "decision-support environment."
        )

        email = st.text_input(
            "Email",
            placeholder="Enter your email"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password"
        )

        login_clicked = st.button(
            "Sign In",
            use_container_width=True,
            type="primary"
        )

        if login_clicked:

            if (
                email in DEMO_USERS
                and password == DEMO_USERS[email]["password"]
            ):

                st.session_state.authenticated = True
                st.session_state.username = email

                st.success("Login successful.")

                st.rerun()

            else:

                st.error(
                    "Invalid email or password."
                )

        st.divider()

        # ----------------------------------------------------
        # GOOGLE LOGIN
        # ----------------------------------------------------

        st.subheader("Alternative Sign-In")

        st.info(
            "Google login can be enabled later using Streamlit "
            "OIDC/Google authentication. The current prototype "
            "uses email and password authentication."
        )

        # ----------------------------------------------------
        # FORGOT PASSWORD
        # ----------------------------------------------------

        if st.button(
            "Forgot Password?"
        ):

            st.session_state.show_forgot_password = True

        if st.session_state.show_forgot_password:

            st.warning(
                "This prototype does not send real password-reset "
                "emails. For the demonstration environment, the "
                "administrator can change the password directly "
                "in DEMO_USERS inside Dashboard.py."
            )

        st.divider()

        st.caption(
            "Prototype / Research Demonstration System"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# 8. AUTHENTICATION GATE
# ============================================================

if not st.session_state.authenticated:

    login_screen()

    st.stop()


# ============================================================
# 9. KAFKA CONFIGURATION
# ============================================================

KAFKA_SERVER = "localhost:9092"

KAFKA_TOPIC = "fraud-predictions"


# ============================================================
# 10. KAFKA CONSUMER
# ============================================================

@st.cache_resource
def create_kafka_consumer():

    try:

        consumer = KafkaConsumer(

            KAFKA_TOPIC,

            bootstrap_servers=KAFKA_SERVER,

            value_deserializer=lambda message:
                json.loads(message.decode("utf-8")),

            auto_offset_reset="latest",

            enable_auto_commit=True,

            group_id="fraud-dashboard-executive",

            consumer_timeout_ms=1000
        )

        return consumer

    except KafkaError as e:

        st.error(
            f"Kafka connection failed: {e}"
        )

        return None

    except Exception as e:

        st.error(
            f"Kafka connection failed: {e}"
        )

        return None


consumer = create_kafka_consumer()


# ============================================================
# 11. READ KAFKA TRANSACTIONS
# ============================================================

def read_kafka_transactions():

    if consumer is None:
        return

    try:

        messages = consumer.poll(
            timeout_ms=1000,
            max_records=100
        )

        existing_ids = {
            t.get("transaction_id")
            for t in st.session_state.transactions
        }

        for _, records in messages.items():

            for message in records:

                transaction = message.value

                if not isinstance(transaction, dict):
                    continue

                transaction_id = transaction.get(
                    "transaction_id"
                )

                if (
                    transaction_id is None
                    or transaction_id not in existing_ids
                ):

                    st.session_state.transactions.append(
                        transaction
                    )

                    existing_ids.add(transaction_id)

    except Exception as e:

        st.warning(
            f"Unable to read Kafka transactions: {e}"
        )


read_kafka_transactions()


# ============================================================
# 12. LOAD MACHINE LEARNING MODEL
# ============================================================

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


@st.cache_resource
def load_prediction_model():

    try:

        model = joblib.load(
            "rf_fraud_model.pkl"
        )

        explainer = shap.TreeExplainer(
            model
        )

        return model, explainer, None

    except Exception as e:

        return None, None, str(e)


pred_model, pred_explainer, model_error = (
    load_prediction_model()
)


# ============================================================
# 13. HELPER FUNCTIONS
# ============================================================

def normalise_probability(value):

    try:

        value = float(value)

        if value > 1:
            value = value / 100

        return max(0, min(1, value))

    except:

        return 0.0


# ------------------------------------------------------------
# Recommended DSS action
# ------------------------------------------------------------

def recommend_action(transaction):

    prediction = transaction.get(
        "prediction",
        "UNKNOWN"
    )

    probability = normalise_probability(
        transaction.get(
            "fraud_probability",
            0
        )
    )

    if probability >= 0.85:

        return (
            "BLOCK / ESCALATE",
            "Very high fraud risk. "
            "Transaction should be investigated and "
            "the relevant account or transaction controls "
            "considered."
        )

    elif probability >= 0.60:

        return (
            "REQUEST CUSTOMER VERIFICATION",
            "Elevated fraud risk. "
            "Additional customer verification is recommended "
            "before completing the transaction."
        )

    elif prediction == "FRAUD":

        return (
            "ESCALATE CASE",
            "The model has classified the transaction as "
            "fraudulent. Human review is recommended."
        )

    else:

        return (
            "APPROVE / MONITOR",
            "Low observed fraud risk. "
            "Transaction can proceed subject to normal controls."
        )


# ------------------------------------------------------------
# Record operational action
# ------------------------------------------------------------

def record_action(
    transaction,
    action,
    analyst
):

    transaction_id = transaction.get(
        "transaction_id",
        "Unknown"
    )

    probability = normalise_probability(
        transaction.get(
            "fraud_probability",
            0
        )
    )

    record = {

        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "transaction_id": transaction_id,

        "fraud_probability": probability,

        "prediction": transaction.get(
            "prediction",
            "UNKNOWN"
        ),

        "recommended_action": action,

        "analyst_action": action,

        "analyst": analyst

    }

    st.session_state.action_log.append(
        record
    )


# ------------------------------------------------------------
# SHAP helper
# ------------------------------------------------------------

def calculate_shap_explanation(
    input_df
):

    if pred_explainer is None:
        return None

    try:

        shap_values = pred_explainer.shap_values(
            input_df
        )

        if isinstance(shap_values, list):

            fraud_shap = np.asarray(
                shap_values[1]
            )[0]

        else:

            shap_values = np.asarray(
                shap_values
            )

            if shap_values.ndim == 3:

                fraud_shap = shap_values[
                    0, :, 1
                ]

            elif shap_values.ndim == 2:

                fraud_shap = shap_values[0]

            else:

                fraud_shap = shap_values

        shap_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Feature_Value":
                    input_df.iloc[0].values,
                "SHAP_Value":
                    fraud_shap
            }
        )

        shap_df["Absolute_SHAP"] = (
            shap_df["SHAP_Value"].abs()
        )

        shap_df = shap_df.sort_values(
            "Absolute_SHAP",
            ascending=False
        )

        shap_df["Direction"] = np.where(
            shap_df["SHAP_Value"] >= 0,
            "Increases Fraud Risk",
            "Reduces Fraud Risk"
        )

        return shap_df

    except Exception as e:

        st.warning(
            f"SHAP explanation unavailable: {e}"
        )

        return None


# ============================================================
# 14. SIDEBAR
# ============================================================

with st.sidebar:

    st.image(
        SHIELD_ICON,
        width=55
    )

    st.title("Fraud DSS")

    st.caption(
        "Interpretable Fraud Detection & "
        "Operational Decision Support"
    )

    st.divider()

    # --------------------------------------------------------
    # User
    # --------------------------------------------------------

    user_information = DEMO_USERS.get(
        st.session_state.username,
        {}
    )

    st.write(
        f"**User:** "
        f"{user_information.get('name', 'User')}"
    )

    st.caption(
        f"Role: "
        f"{user_information.get('role', 'Analyst')}"
    )

    st.divider()

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    current_page = st.radio(

        "System Navigation",

        [

            "Welcome",

            "Executive Overview",

            "Historical / Batch Analytics",

            "Historical Model Analysis",

            "Predict Fraud",

            "Real-Time Fraud Detection",

            "Transaction Explorer",

            "Operational Decisions",

            "Trends & Monitoring",

            "Threshold Tuner",

            "AI Decision Assistant",

            "Adaptive Monitoring",

            "About the Framework"

        ]
    )

    st.divider()

    # --------------------------------------------------------
    # System status
    # --------------------------------------------------------

    st.caption("SYSTEM STATUS")

    if consumer is not None:

        st.success(
            "Kafka Online",
            icon=":material/check_circle:"
        )

    else:

        st.error(
            "Kafka Offline",
            icon=":material/error:"
        )

    st.caption(
        f"Kafka Topic: {KAFKA_TOPIC}"
    )

    st.divider()

    if st.button(
        "Refresh Dashboard",
        use_container_width=True
    ):

        st.rerun()

    if st.button(
        "Sign Out",
        use_container_width=True
    ):

        st.session_state.authenticated = False
        st.session_state.username = ""

        st.rerun()


# ============================================================
# 15. WELCOME
# ============================================================

if current_page == "Welcome":

    st.image(
        SHIELD_ICON,
        width=90
    )

    st.markdown(
        """
        <div class="main-title">
            WELCOME TO FRAUD DSS
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            Interpretable Fraud Detection & Operational
            Decision Support
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.success(
        f"Welcome back, "
        f"{user_information.get('name', 'User')}."
    )

    st.write(
        """
        This Decision Support System integrates historical
        fraud analytics, machine learning, real-time transaction
        streaming, explainable AI and operational decision
        support within a single environment.
        """
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.image(
            MODEL_ICON,
            width=55
        )

        st.subheader(
            "Machine Learning"
        )

        st.caption(
            "Random Forest fraud classification."
        )

    with c2:

        st.image(
            LIVE_ICON,
            width=55
        )

        st.subheader(
            "Real-Time Stream"
        )

        st.caption(
            "Kafka-based transaction ingestion."
        )

    with c3:

        st.image(
            XAI_ICON,
            width=55
        )

        st.subheader(
            "Explainable AI"
        )

        st.caption(
            "SHAP-based transaction explanations."
        )

    with c4:

        st.image(
            OVERVIEW_ICON,
            width=55
        )

        st.subheader(
            "Decision Support"
        )

        st.caption(
            "Evidence-based operational recommendations."
        )

    st.divider()

    st.info(
        """
        **Prototype note:** The DSS supports human decision-making.
        Recommended actions such as verification, blocking and
        escalation are simulated within this research prototype
        and are not connected to real banking systems.
        """
    )


# ============================================================
# 16. EXECUTIVE OVERVIEW
# ============================================================

elif current_page == "Executive Overview":

    st.image(
        OVERVIEW_ICON,
        width=65
    )

    st.title(
        "Executive Overview"
    )

    st.caption(
        "High-level management view of fraud risk, "
        "transaction activity and operational response."
    )

    st.divider()

    transactions = (
        st.session_state.transactions
    )

    total = len(transactions)

    fraud = sum(
        t.get("prediction") == "FRAUD"
        for t in transactions
    )

    legitimate = sum(
        t.get("prediction") == "LEGITIMATE"
        for t in transactions
    )

    fraud_rate = (
        fraud / total * 100
        if total > 0
        else 0
    )

    total_value = sum(
        float(t.get("Amount", 0) or 0)
        for t in transactions
    )

    fraud_value = sum(
        float(t.get("Amount", 0) or 0)
        for t in transactions
        if t.get("prediction") == "FRAUD"
    )

    # ========================================================
    # KPI CARDS
    # ========================================================

    st.subheader(
        "Executive KPIs"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "Transactions",
            f"{total:,}"
        )

    with c2:

        st.metric(
            "Fraud Detected",
            f"{fraud:,}"
        )

    with c3:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )

    with c4:

        st.metric(
            "Transaction Value",
            f"{total_value:,.2f}"
        )

    with c5:

        st.metric(
            "Risk Value",
            f"{fraud_value:,.2f}"
        )

    st.divider()

    # ========================================================
    # EXECUTIVE CHARTS
    # ========================================================

    if transactions:

        df = pd.DataFrame(
            transactions
        )

        chart_col1, chart_col2 = st.columns(2)

        # ----------------------------------------------------
        # Fraud vs Legitimate
        # ----------------------------------------------------

        with chart_col1:

            st.subheader(
                "Transaction Classification"
            )

            classification_df = pd.DataFrame(
                {
                    "Classification":
                        ["Legitimate", "Fraud"],
                    "Transactions":
                        [legitimate, fraud]
                }
            )

            st.bar_chart(
                classification_df.set_index(
                    "Classification"
                )
            )

        # ----------------------------------------------------
        # Fraud Probability
        # ----------------------------------------------------

        with chart_col2:

            st.subheader(
                "Fraud Risk Distribution"
            )

            if "fraud_probability" in df.columns:

                risk_df = pd.to_numeric(
                    df["fraud_probability"],
                    errors="coerce"
                )

                risk_df = risk_df.apply(
                    normalise_probability
                )

                bins = pd.cut(
                    risk_df,
                    bins=[
                        -0.01,
                        0.20,
                        0.40,
                        0.60,
                        0.80,
                        1.0
                    ],
                    labels=[
                        "0-20%",
                        "20-40%",
                        "40-60%",
                        "60-80%",
                        "80-100%"
                    ]
                )

                risk_counts = (
                    bins.value_counts()
                    .sort_index()
                )

                st.bar_chart(
                    risk_counts
                )

        st.divider()

        # ====================================================
        # MANAGEMENT INSIGHT
        # ====================================================

        st.subheader(
            "Executive Risk Insight"
        )

        if fraud_rate >= 5:

            st.warning(
                f"Fraud activity currently represents "
                f"{fraud_rate:.2f}% of observed transactions. "
                f"Management attention and investigation "
                f"capacity may need to be increased."
            )

        elif fraud_rate > 0:

            st.info(
                f"The current stream contains "
                f"{fraud} flagged transaction(s), "
                f"representing {fraud_rate:.2f}% of observed "
                f"transactions."
            )

        else:

            st.success(
                "No fraudulent transactions have been "
                "identified in the currently observed stream."
            )

        st.divider()

        # ====================================================
        # RECENT HIGH-RISK TRANSACTIONS
        # ====================================================

        st.subheader(
            "Recent High-Risk Transactions"
        )

        if "fraud_probability" in df.columns:

            df["risk_numeric"] = pd.to_numeric(
                df["fraud_probability"],
                errors="coerce"
            ).apply(
                normalise_probability
            )

            high_risk = df[
                df["risk_numeric"] >= 0.60
            ].sort_values(
                "risk_numeric",
                ascending=False
            )

            if high_risk.empty:

                st.info(
                    "No high-risk transactions currently available."
                )

            else:

                display_columns = [
                    c for c in [
                        "transaction_id",
                        "prediction",
                        "fraud_probability",
                        "Amount",
                        "timestamp"
                    ]
                    if c in high_risk.columns
                ]

                st.dataframe(
                    high_risk[
                        display_columns
                    ].head(10),
                    use_container_width=True,
                    hide_index=True
                )

    else:

        st.info(
            "No live transactions have been received yet. "
            "Start the Kafka/Spark pipeline to populate the "
            "executive dashboard."
        )


# ============================================================
# 17. HISTORICAL / BATCH ANALYTICS
# ============================================================

elif current_page == "Historical / Batch Analytics":

    st.image(
        OVERVIEW_ICON,
        width=65
    )

    st.title(
        "Historical / Batch Analytics"
    )

    st.write(
        """
        This dashboard represents the analytical layer for
        historical and batch-processed transaction data.
        It is intentionally separated from the real-time
        transaction monitoring dashboard.
        """
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload historical transaction results",
        type=["csv"]
    )

    if uploaded_file is not None:

        historical_df = pd.read_csv(
            uploaded_file
        )

        st.session_state.historical_df = (
            historical_df
        )

    historical_df = (
        st.session_state.historical_df
    )

    if historical_df is not None:

        df = historical_df.copy()

        st.subheader(
            "Historical Processing KPIs"
        )

        total_hist = len(df)

        if "prediction" in df.columns:

            fraud_hist = int(
                (
                    df["prediction"]
                    .astype(str)
                    .str.upper()
                    == "FRAUD"
                ).sum()
            )

        elif "Class" in df.columns:

            fraud_hist = int(
                df["Class"].sum()
            )

        else:

            fraud_hist = 0

        fraud_rate_hist = (
            fraud_hist / total_hist * 100
            if total_hist
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Records Processed",
                f"{total_hist:,}"
            )

        with c2:
            st.metric(
                "Fraud Cases",
                f"{fraud_hist:,}"
            )

        with c3:
            st.metric(
                "Fraud Rate",
                f"{fraud_rate_hist:.2f}%"
            )

        with c4:

            if "Amount" in df.columns:

                amount_value = pd.to_numeric(
                    df["Amount"],
                    errors="coerce"
                ).sum()

                st.metric(
                    "Transaction Value",
                    f"{amount_value:,.2f}"
                )

            else:

                st.metric(
                    "Transaction Value",
                    "N/A"
                )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Historical Classification"
            )

            if "prediction" in df.columns:

                counts = (
                    df["prediction"]
                    .value_counts()
                )

                st.bar_chart(
                    counts
                )

            elif "Class" in df.columns:

                counts = (
                    df["Class"]
                    .map(
                        {
                            0: "LEGITIMATE",
                            1: "FRAUD"
                        }
                    )
                    .value_counts()
                )

                st.bar_chart(
                    counts
                )

        with col2:

            st.subheader(
                "Transaction Amount Distribution"
            )

            if "Amount" in df.columns:

                amount_series = pd.to_numeric(
                    df["Amount"],
                    errors="coerce"
                ).dropna()

                st.line_chart(
                    amount_series.reset_index(
                        drop=True
                    ).head(500)
                )

        st.divider()

        st.subheader(
            "Historical Data Preview"
        )

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

    else:

        st.info(
            "Upload a historical/batch CSV file to display "
            "historical processing metrics and charts."
        )


# ============================================================
# 18. HISTORICAL MODEL ANALYSIS
# ============================================================

elif current_page == "Historical Model Analysis":

    st.image(
        MODEL_ICON,
        width=65
    )

    st.title(
        "Historical Model Analysis"
    )

    st.write(
        """
        Historical model evaluation provides the evidence base
        for understanding how well the trained fraud detection
        model performs before deployment into the streaming
        environment.
        """
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "ROC-AUC",
            "0.961"
        )

    with c2:
        st.metric(
            "PR-AUC",
            "0.813"
        )

    with c3:
        st.metric(
            "Fraud Precision",
            "0.90"
        )

    with c4:
        st.metric(
            "Fraud Recall",
            "0.77"
        )

    st.divider()

    st.subheader(
        "Model Performance Interpretation"
    )

    st.write(
        """
        The model demonstrates strong discrimination between
        legitimate and fraudulent transactions. Precision and
        recall are presented separately because fraud detection
        involves a trade-off between identifying fraudulent
        activity and limiting false alarms.
        """
    )

    st.divider()

    st.subheader(
        "Global Model Interpretability"
    )

    if model_error:

        st.warning(
            f"Model/SHAP files could not be loaded: "
            f"{model_error}"
        )

    else:

        st.success(
            "Random Forest model and SHAP explainer loaded."
        )

        st.info(
            """
            SHAP provides global interpretability by showing
            which features have the greatest influence on
            predictions across the model.
            """
        )

    st.divider()

    st.subheader(
        "Model Metrics"
    )

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "ROC-AUC",
                "PR-AUC",
                "Fraud Precision",
                "Fraud Recall",
                "Fraud F1-score"
            ],
            "Value": [
                0.961,
                0.813,
                0.90,
                0.77,
                0.83
            ]
        }
    )

    st.bar_chart(
        metrics_df.set_index(
            "Metric"
        )
    )


# ============================================================
# 19. PREDICT FRAUD
# ============================================================

elif current_page == "Predict Fraud":

    st.image(
        XAI_ICON,
        width=65
    )

    st.title(
        "Predict Fraud"
    )

    st.write(
        """
        Perform an individual or batch prediction using the
        trained Random Forest model and examine the explanatory
        factors behind the prediction.
        """
    )

    st.divider()

    if pred_model is None:

        st.error(
            "rf_fraud_model.pkl could not be loaded."
        )

    else:

        tab_manual, tab_batch = st.tabs(
            [
                "Manual Transaction",
                "Batch Prediction"
            ]
        )

        # ====================================================
        # MANUAL PREDICTION
        # ====================================================

        with tab_manual:

            st.subheader(
                "Enter Transaction Details"
            )

            st.caption(
                "V1-V28 are PCA-transformed features."
            )

            with st.form(
                "manual_prediction_form"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    input_time = st.number_input(
                        "Time",
                        value=0.0
                    )

                with col2:

                    input_amount = st.number_input(
                        "Amount",
                        value=0.0
                    )

                st.markdown(
                    "**V1 - V28**"
                )

                v_inputs = {}

                cols = st.columns(4)

                for i in range(1, 29):

                    col = cols[
                        (i - 1) % 4
                    ]

                    with col:

                        v_inputs[
                            f"V{i}"
                        ] = st.number_input(
                            f"V{i}",
                            value=0.0,
                            key=f"manual_v_{i}"
                        )

                submit = st.form_submit_button(
                    "Predict Transaction",
                    use_container_width=True,
                    type="primary"
                )

            if submit:

                row = {
                    "Time": input_time
                }

                row.update(
                    v_inputs
                )

                row["Amount"] = input_amount

                input_df = pd.DataFrame(
                    [row],
                    columns=FEATURE_COLUMNS
                )

                prediction = (
                    pred_model.predict(
                        input_df
                    )[0]
                )

                probability = (
                    pred_model.predict_proba(
                        input_df
                    )[0][1]
                )

                result = (
                    "FRAUD"
                    if prediction == 1
                    else "LEGITIMATE"
                )

                st.divider()

                if result == "FRAUD":

                    st.error(
                        f"FRAUD DETECTED — "
                        f"Probability: "
                        f"{probability:.2%}"
                    )

                else:

                    st.success(
                        f"LEGITIMATE — "
                        f"Fraud Probability: "
                        f"{probability:.2%}"
                    )

                shap_df = (
                    calculate_shap_explanation(
                        input_df
                    )
                )

                if shap_df is not None:

                    st.subheader(
                        "Top Risk Factors"
                    )

                    st.dataframe(
                        shap_df.head(10)[
                            [
                                "Feature",
                                "Feature_Value",
                                "SHAP_Value",
                                "Direction"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

        # ====================================================
        # BATCH PREDICTION
        # ====================================================

        with tab_batch:

            uploaded_file = st.file_uploader(
                "Upload transaction CSV",
                type="csv",
                key="batch_prediction"
            )

            if uploaded_file:

                batch_df = pd.read_csv(
                    uploaded_file
                )

                missing = (
                    set(FEATURE_COLUMNS)
                    - set(batch_df.columns)
                )

                if missing:

                    st.error(
                        f"Missing columns: "
                        f"{sorted(missing)}"
                    )

                else:

                    X = batch_df[
                        FEATURE_COLUMNS
                    ]

                    predictions = (
                        pred_model.predict(X)
                    )

                    probabilities = (
                        pred_model.predict_proba(X)[
                            :, 1
                        ]
                    )

                    results_df = batch_df.copy()

                    results_df[
                        "prediction"
                    ] = np.where(
                        predictions == 1,
                        "FRAUD",
                        "LEGITIMATE"
                    )

                    results_df[
                        "fraud_probability"
                    ] = probabilities

                    total_batch = len(
                        results_df
                    )

                    fraud_batch = int(
                        (
                            predictions == 1
                        ).sum()
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "Records",
                            total_batch
                        )

                    with c2:

                        st.metric(
                            "Fraud",
                            fraud_batch
                        )

                    with c3:

                        st.metric(
                            "Fraud Rate",
                            f"{(
                                fraud_batch
                                / total_batch
                                * 100
                            ):.2f}%"
                            if total_batch
                            else "0%"
                        )

                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )

                    csv_data = (
                        results_df
                        .to_csv(index=False)
                        .encode("utf-8")
                    )

                    st.download_button(
                        "Download Results",
                        data=csv_data,
                        file_name="fraud_predictions_results.csv",
                        mime="text/csv"
                    )


# ============================================================
# 20. REAL-TIME FRAUD DETECTION
# ============================================================

elif current_page == "Real-Time Fraud Detection":

    st.image(
        LIVE_ICON,
        width=65
    )

    st.title(
        "Real-Time Fraud Detection"
    )

    st.write(
        """
        Live monitoring of transactions processed through
        Kafka and the real-time fraud detection pipeline.
        """
    )

    st.divider()

    transactions = (
        st.session_state.transactions
    )

    total = len(transactions)

    fraud = sum(
        t.get("prediction") == "FRAUD"
        for t in transactions
    )

    fraud_rate = (
        fraud / total * 100
        if total
        else 0
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Live Transactions",
            total
        )

    with c2:

        st.metric(
            "Fraud Detected",
            fraud
        )

    with c3:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )

    st.divider()

    if not transactions:

        st.info(
            "Waiting for transactions from Kafka..."
        )

    else:

        latest = transactions[-1]

        probability = normalise_probability(
            latest.get(
                "fraud_probability",
                0
            )
        )

        prediction = latest.get(
            "prediction",
            "UNKNOWN"
        )

        transaction_id = latest.get(
            "transaction_id",
            "Unknown"
        )

        if prediction == "FRAUD":

            st.error(
                f"FRAUD DETECTED | "
                f"Transaction: {transaction_id} | "
                f"Risk: {probability:.2%}"
            )

        else:

            st.success(
                f"LEGITIMATE | "
                f"Transaction: {transaction_id} | "
                f"Risk: {probability:.2%}"
            )

        st.divider()

        st.subheader(
            "Latest Transaction"
        )

        st.json(
            latest
        )

        st.divider()

        # ====================================================
        # DSS RECOMMENDATION
        # ====================================================

        action, explanation = (
            recommend_action(
                latest
            )
        )

        st.subheader(
            "DSS Recommendation"
        )

        st.info(
            f"**Recommended Action:** {action}\n\n"
            f"{explanation}"
        )

        st.caption(
            "Recommendation is generated for decision-support "
            "purposes and requires human review."
        )

        # ====================================================
        # OPERATIONAL BUTTONS
        # ====================================================

        st.subheader(
            "Operational Response"
        )

        a1, a2, a3, a4 = st.columns(4)

        with a1:

            if st.button(
                "Approve",
                use_container_width=True
            ):

                record_action(
                    latest,
                    "APPROVE",
                    st.session_state.username
                )

                st.success(
                    "Approval recorded in DSS action log."
                )

        with a2:

            if st.button(
                "Request Verification",
                use_container_width=True
            ):

                record_action(
                    latest,
                    "REQUEST CUSTOMER VERIFICATION",
                    st.session_state.username
                )

                st.warning(
                    "Verification request recorded."
                )

        with a3:

            if st.button(
                "Block",
                use_container_width=True
            ):

                record_action(
                    latest,
                    "BLOCK TRANSACTION / ACCOUNT",
                    st.session_state.username
                )

                st.error(
                    "Blocking recommendation recorded. "
                    "No real account was blocked."
                )

        with a4:

            if st.button(
                "Escalate",
                use_container_width=True
            ):

                record_action(
                    latest,
                    "ESCALATE CASE",
                    st.session_state.username
                )

                st.warning(
                    "Escalation recorded in DSS action log."
                )

        # ====================================================
        # SHAP
        # ====================================================

        if latest.get(
            "top_shap_features"
        ):

            st.divider()

            st.subheader(
                "Explainable AI — Top Risk Factors"
            )

            shap_data = latest[
                "top_shap_features"
            ]

            st.dataframe(
                pd.DataFrame(
                    shap_data
                ),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 21. TRANSACTION EXPLORER
# ============================================================

elif current_page == "Transaction Explorer":

    st.image(
        TRANSACTION_ICON,
        width=65
    )

    st.title(
        "Transaction Explorer"
    )

    st.write(
        """
        Trace and investigate individual transactions using
        their transaction ID.
        """
    )

    st.divider()

    query = st.text_input(
        "Transaction ID",
        placeholder="Enter transaction ID..."
    )

    transactions = (
        st.session_state.transactions
    )

    if query:

        matches = [

            t for t in transactions

            if query.lower()
            in str(
                t.get(
                    "transaction_id",
                    ""
                )
            ).lower()

        ]

        if matches:

            transaction = matches[0]

            st.success(
                f"{len(matches)} transaction(s) found."
            )

            st.json(
                transaction
            )

            st.divider()

            probability = normalise_probability(
                transaction.get(
                    "fraud_probability",
                    0
                )
            )

            action, explanation = (
                recommend_action(
                    transaction
                )
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Prediction",
                    transaction.get(
                        "prediction",
                        "UNKNOWN"
                    )
                )

            with c2:

                st.metric(
                    "Fraud Probability",
                    f"{probability:.2%}"
                )

            with c3:

                st.metric(
                    "Recommended Action",
                    action
                )

            st.info(
                explanation
            )

            st.divider()

            st.subheader(
                "Transaction-Level Decision Support"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                if st.button(
                    "Approve Transaction",
                    use_container_width=True
                ):

                    record_action(
                        transaction,
                        "APPROVE",
                        st.session_state.username
                    )

                    st.success(
                        "Approval recorded."
                    )

            with c2:

                if st.button(
                    "Request Verification",
                    use_container_width=True
                ):

                    record_action(
                        transaction,
                        "REQUEST CUSTOMER VERIFICATION",
                        st.session_state.username
                    )

                    st.warning(
                        "Verification request recorded."
                    )

            with c3:

                if st.button(
                    "Block",
                    use_container_width=True
                ):

                    record_action(
                        transaction,
                        "BLOCK TRANSACTION / ACCOUNT",
                        st.session_state.username
                    )

                    st.error(
                        "Blocking recommendation recorded. "
                        "No real account was blocked."
                    )

            with c4:

                if st.button(
                    "Escalate",
                    use_container_width=True
                ):

                    record_action(
                        transaction,
                        "ESCALATE CASE",
                        st.session_state.username
                    )

                    st.warning(
                        "Case escalation recorded."
                    )

            if transaction.get(
                "top_shap_features"
            ):

                st.divider()

                st.subheader(
                    "Explainable AI"
                )

                st.dataframe(
                    pd.DataFrame(
                        transaction[
                            "top_shap_features"
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.warning(
                "No matching transaction found."
            )

    elif transactions:

        st.subheader(
            "Available Transactions"
        )

        st.dataframe(
            pd.DataFrame(
                transactions
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No transactions are currently available."
        )


# ============================================================
# 22. OPERATIONAL DECISIONS
# ============================================================

elif current_page == "Operational Decisions":

    st.image(
        OVERVIEW_ICON,
        width=65
    )

    st.title(
        "Operational Decision Support"
    )

    st.write(
        """
        This section records the operational responses selected
        by the analyst after reviewing model predictions,
        fraud probability and explainable AI evidence.
        """
    )

    st.divider()

    actions = st.session_state.action_log

    if not actions:

        st.info(
            "No operational decisions have been recorded yet."
        )

    else:

        action_df = pd.DataFrame(
            actions
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Actions Recorded",
                len(action_df)
            )

        with c2:

            st.metric(
                "Approvals",
                int(
                    (
                        action_df[
                            "analyst_action"
                        ]
                        == "APPROVE"
                    ).sum()
                )
            )

        with c3:

            st.metric(
                "Verification Requests",
                int(
                    (
                        action_df[
                            "analyst_action"
                        ]
                        == "REQUEST CUSTOMER VERIFICATION"
                    ).sum()
                )
            )

        with c4:

            st.metric(
                "Escalations",
                int(
                    (
                        action_df[
                            "analyst_action"
                        ]
                        == "ESCALATE CASE"
                    ).sum()
                )
            )

        st.divider()

        st.subheader(
            "Decision Distribution"
        )

        counts = (
            action_df[
                "analyst_action"
            ]
            .value_counts()
        )

        st.bar_chart(
            counts
        )

        st.divider()

        st.subheader(
            "Operational Action Log"
        )

        st.dataframe(
            action_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = (
            action_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Decision Log",
            data=csv_data,
            file_name="dss_operational_decision_log.csv",
            mime="text/csv"
        )


# ============================================================
# 23. TRENDS & MONITORING
# ============================================================

elif current_page == "Trends & Monitoring":

    st.image(
        TREND_ICON,
        width=65
    )

    st.title(
        "Trends & Monitoring"
    )

    st.write(
        """
        Monitor transaction volumes, fraud classifications
        and changing risk patterns.
        """
    )

    st.divider()

    transactions = (
        st.session_state.transactions
    )

    if transactions:

        df = pd.DataFrame(
            transactions
        )

        # ----------------------------------------------------
        # Timestamp trend
        # ----------------------------------------------------

        if "timestamp" in df.columns:

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce"
            )

            df = df.dropna(
                subset=["timestamp"]
            )

            if not df.empty:

                volume = (
                    df.set_index(
                        "timestamp"
                    )
                    .resample("1min")
                    .size()
                )

                st.subheader(
                    "Transaction Volume Over Time"
                )

                st.line_chart(
                    volume
                )

        # ----------------------------------------------------
        # Fraud trend
        # ----------------------------------------------------

        if (
            "timestamp" in df.columns
            and "prediction" in df.columns
        ):

            fraud_df = df.copy()

            fraud_df["fraud_flag"] = (
                fraud_df[
                    "prediction"
                ]
                .astype(str)
                .str.upper()
                .eq("FRAUD")
                .astype(int)
            )

            fraud_trend = (
                fraud_df.set_index(
                    "timestamp"
                )[
                    "fraud_flag"
                ]
                .resample("1min")
                .sum()
            )

            st.subheader(
                "Fraud Detection Trend"
            )

            st.line_chart(
                fraud_trend
            )

    else:

        st.info(
            "Waiting for transaction data..."
        )


# ============================================================
# 24. THRESHOLD TUNER
# ============================================================

elif current_page == "Threshold Tuner":

    st.image(
        THRESHOLD_ICON,
        width=65
    )

    st.title(
        "Fraud Detection Threshold"
    )

    st.write(
        """
        Explore how changing the probability threshold changes
        operational classification.
        """
    )

    st.divider()

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01
    )

    st.metric(
        "Selected Threshold",
        f"{threshold:.2f}"
    )

    st.divider()

    st.info(
        """
        A lower threshold increases sensitivity and may detect
        more fraudulent transactions, but may also increase
        false positives.

        A higher threshold can reduce false alarms but may
        increase the possibility of missed fraud.

        Threshold selection should therefore consider validation
        performance, operational capacity and the relative cost
        of false positives and false negatives.
        """
    )


# ============================================================
# 25. AI DECISION ASSISTANT
# ============================================================

elif current_page == "AI Decision Assistant":

    st.image(
        AI_ASSISTANT_ICON,
        width=65
    )

    st.title(
        "AI Decision Assistant"
    )

    st.write(
        """
        An interface for supporting analysts in interpreting
        fraud predictions, transaction risk and model
        explanations.
        """
    )

    st.divider()

    question = st.chat_input(
        "Ask about a transaction or model result..."
    )

    if question:

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            st.info(
                """
                The decision-assistant interface is available
                for integration with transaction records,
                SHAP explanations and operational decision logs.

                Current implementation provides the dashboard
                interface; a production-grade language model
                integration would be a separate component.
                """
            )


# ============================================================
# 26. ADAPTIVE MONITORING
# ============================================================

elif current_page == "Adaptive Monitoring":

    st.image(
        MODEL_ICON,
        width=65
    )

    st.title(
        "Adaptive Monitoring"
    )

    st.write(
        """
        Monitoring indicators that support adaptation of the
        fraud detection framework as transaction behaviour and
        fraud patterns change.
        """
    )

    st.divider()

    transactions = (
        st.session_state.transactions
    )

    total = len(transactions)

    fraud = sum(
        t.get("prediction") == "FRAUD"
        for t in transactions
    )

    fraud_rate = (
        fraud / total * 100
        if total
        else 0
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Observed Transactions",
            total
        )

    with c2:

        st.metric(
            "Observed Fraud Cases",
            fraud
        )

    with c3:

        st.metric(
            "Current Fraud Rate",
            f"{fraud_rate:.2f}%"
        )

    st.divider()

    st.subheader(
        "Adaptation Logic"
    )

    st.write(
        """
        The framework is designed to support adaptation through
        continuous monitoring of incoming transactions, fraud
        patterns, model performance and operational outcomes.

        When material changes are identified, historical data
        can be incorporated into subsequent model development,
        validation and threshold reassessment.
        """
    )

    st.divider()

    st.subheader(
        "Current Prototype Adaptation Status"
    )

    st.info(
        """
        **Monitoring:** Active

        **Real-time ingestion:** Kafka

        **Streaming processing:** Apache Spark

        **Model explanation:** SHAP

        **Threshold reassessment:** Available

        **Automatic model retraining:** Not enabled in this
        prototype

        **Human review:** Required for operational decisions
        """
    )

    st.warning(
        """
        This distinction is important: the prototype demonstrates
        an adaptive framework and adaptation workflow rather than
        claiming that the Random Forest model automatically
        retrains itself after every transaction.
        """
    )


# ============================================================
# 27. ABOUT THE FRAMEWORK
# ============================================================

elif current_page == "About the Framework":

    st.image(
        INFO_ICON,
        width=65
    )

    st.title(
        "About the Framework"
    )

    st.markdown(
        """
        ### Development of Adaptive and Interpretable Machine
        Learning Framework for Real-Time Fraud Detection and
        Operational Decision Support in Digital Banking

        **Developed by: Adeseye Samuel Ademola**
        """
    )

    st.divider()

    st.write(
        """
        The framework integrates historical data analytics,
        machine learning, Apache Kafka, Apache Spark, Explainable
        Artificial Intelligence and operational decision support.
        """
    )

    st.divider()

    st.subheader(
        "Framework Components"
    )

    components = pd.DataFrame(
        {
            "Component": [
                "Historical Data",
                "Machine Learning",
                "Apache Kafka",
                "Apache Spark",
                "Explainable AI",
                "Decision Support",
                "Streamlit",
                "Docker"
            ],

            "Purpose": [

                "Model development and historical analysis",

                "Fraud classification",

                "Real-time transaction event streaming",

                "Streaming/batch processing",

                "Model interpretation",

                "Operational recommendations",

                "Interactive DSS interface",

                "Environment/containerisation"
            ]
        }
    )

    st.dataframe(
        components,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Operational Decision Logic"
    )

    st.write(
        """
        The DSS combines model prediction, fraud probability,
        transaction information and explainability evidence to
        support operational recommendations.

        Possible recommendations include:

        • Approve / Monitor

        • Request Customer Verification

        • Escalate Case

        • Block Transaction / Account

        The recommendations are intended to improve consistency
        and evidence-based decision-making rather than replace
        human judgement.
        """
    )

    st.divider()

    st.subheader(
        "Framework Adaptability"
    )

    st.write(
        """
        Adaptability is supported through monitoring of incoming
        transactions, fraud rates, model performance, thresholds
        and changing transaction behaviour. These observations
        can inform subsequent retraining, validation and
        threshold adjustment.
        """
    )

    st.divider()

    st.subheader(
        "Prototype Scope"
    )

    st.info(
        """
        This system is a research prototype/reference
        implementation. It demonstrates how the components of an
        adaptive, interpretable fraud detection and operational
        decision-support framework can work together.

        It is not presented as a production banking deployment.
        """
    )

    st.divider()

    st.caption(
        "Interpretable Fraud Detection & Operational "
        "Decision Support System"
    )

    st.caption(
        "Developed by Adeseye Samuel Ademola"
    )