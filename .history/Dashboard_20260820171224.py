# ============================================================
# XAI DSS
# STREAMLIT DECISION SUPPORT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime


# ============================================================
# 1. PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="XAI Fraud Detection DSS",
    page_icon=":material/shield:",
    layout="wide"
)


# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

div[data-testid="stMetric"] {
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "transactions" not in st.session_state:
    st.session_state.transactions = []

if "last_transaction" not in st.session_state:
    st.session_state.last_transaction = None


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.title("🛡️ XAI DSS")

    st.caption(
        "Explainable Fraud Detection "
        "Decision Support System"
    )

    st.divider()

    current_page = st.radio(
        "System Navigation",
        [
            "Executive Overview",
            "Fraud Detection",
            "Model Performance & XAI",
            "Transaction Explorer",
            "Trends & Monitoring",
            "Threshold Tuner",
            "AI Assistant",
            "About"
        ]
    )

    st.divider()

    st.caption("System Status")

    st.success(
        "System Online",
        icon=":material/sensors:"
    )

    if st.button(
        "Refresh Dashboard",
        icon=":material/refresh:",
        use_container_width=True
    ):
        st.rerun()


# ============================================================
# 5. EXECUTIVE OVERVIEW
# ============================================================

if current_page == "Executive Overview":

    st.title(
        "Executive Overview",
        icon=":material/dashboard:"
    )

    st.markdown(
        "Real-time overview of transaction activity, "
        "fraud detection and operational risk."
    )

    st.divider()

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    transactions = st.session_state.transactions

    total_transactions = len(transactions)

    fraud_transactions = sum(
        1
        for t in transactions
        if t.get("prediction") == "FRAUD"
    )

    legitimate_transactions = sum(
        1
        for t in transactions
        if t.get("prediction") == "LEGITIMATE"
    )

    fraud_rate = (
        fraud_transactions / total_transactions * 100
        if total_transactions > 0
        else 0
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Transactions Processed",
            total_transactions,
            icon=":material/swap_horiz:"
        )

    with col2:

        st.metric(
            "Fraud Detected",
            fraud_transactions,
            icon=":material/shield_check:"
        )

    with col3:

        st.metric(
            "Legitimate Transactions",
            legitimate_transactions,
            icon=":material/check_circle:"
        )

    with col4:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%",
            icon=":material/warning:"
        )

    st.divider()

    # --------------------------------------------------------
    # LIVE STATUS
    # --------------------------------------------------------

    st.subheader(
        "Live Transaction Monitoring",
        divider="gray"
    )

    if transactions:

        latest = transactions[-1]

        result = latest.get(
            "prediction",
            "UNKNOWN"
        )

        probability = latest.get(
            "fraud_probability",
            0
        )

        if result == "FRAUD":

            st.error(
                f"Fraud detected — "
                f"Probability: {probability:.2%}",
                icon=":material/gpp_bad:"
            )

        else:

            st.success(
                f"Latest transaction classified as "
                f"LEGITIMATE — "
                f"Fraud probability: {probability:.2%}",
                icon=":material/verified_user:"
            )

    else:

        st.info(
            "Waiting for Kafka transaction data...",
            icon=":material/hourglass_empty:"
        )


# ============================================================
# 6. FRAUD DETECTION
# ============================================================

elif current_page == "Fraud Detection":

    st.title(
        "Fraud Detection",
        icon=":material/shield_check:"
    )

    st.markdown(
        "Real-time fraud classification generated by "
        "the trained Random Forest model."
    )

    st.divider()

    transactions = st.session_state.transactions

    if not transactions:

        st.info(
            "No transaction predictions available yet."
        )

    else:

        fraud_transactions = [
            t for t in transactions
            if t.get("prediction") == "FRAUD"
        ]

        if fraud_transactions:

            st.error(
                f"{len(fraud_transactions)} "
                f"fraud transaction(s) detected.",
                icon=":material/warning:"
            )

            fraud_df = pd.DataFrame(
                fraud_transactions
            )

            columns_to_show = [
                column
                for column in [
                    "transaction_id",
                    "prediction",
                    "fraud_probability",
                    "timestamp"
                ]
                if column in fraud_df.columns
            ]

            st.dataframe(
                fraud_df[columns_to_show],
                use_container_width=True
            )

        else:

            st.success(
                "No fraudulent transactions detected "
                "in the current stream.",
                icon=":material/check_circle:"
            )


# ============================================================
# 7. MODEL PERFORMANCE & XAI
# ============================================================

elif current_page == "Model Performance & XAI":

    st.title(
        "Model Performance & Explainability",
        icon=":material/psychology:"
    )

    st.markdown(
        "Model evaluation metrics and SHAP-based "
        "explanations of individual predictions."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    st.subheader(
        "Model Performance",
        divider="gray"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "ROC-AUC",
            "0.961"
        )

    with col2:

        st.metric(
            "PR-AUC",
            "0.813"
        )

    with col3:

        st.metric(
            "Fraud Precision",
            "0.90"
        )

    with col4:

        st.metric(
            "Fraud Recall",
            "0.77"
        )

    st.divider()

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    st.subheader(
        "Explainable AI (SHAP)",
        divider="gray"
    )

    transactions = st.session_state.transactions

    if transactions:

        latest = transactions[-1]

        shap_values = latest.get(
            "shap_values"
        )

        if shap_values:

            shap_df = pd.DataFrame(
                shap_values,
                columns=[
                    "Feature",
                    "SHAP Value"
                ]
            )

            shap_df["Absolute Impact"] = (
                shap_df["SHAP Value"]
                .abs()
            )

            shap_df = shap_df.sort_values(
                "Absolute Impact",
                ascending=False
            )

            st.bar_chart(
                shap_df.head(10)
                .set_index("Feature")[
                    "SHAP Value"
                ]
            )

            st.dataframe(
                shap_df.head(10),
                use_container_width=True
            )

        else:

            st.info(
                "SHAP explanation will appear "
                "when the XAI pipeline sends explanation data."
            )

    else:

        st.info(
            "Waiting for a transaction prediction "
            "to generate SHAP explanation."
        )


# ============================================================
# 8. TRANSACTION EXPLORER
# ============================================================

elif current_page == "Transaction Explorer":

    st.title(
        "Transaction Explorer",
        icon=":material/manage_search:"
    )

    st.markdown(
        "Search and inspect individual transactions "
        "processed by the fraud detection system."
    )

    st.divider()

    query = st.text_input(
        "Transaction ID",
        placeholder="Enter transaction ID...",
        icon=":material/search:"
    )

    transactions = st.session_state.transactions

    if query:

        matches = [
            t
            for t in transactions
            if query.lower()
            in str(
                t.get(
                    "transaction_id",
                    ""
                )
            ).lower()
        ]

        if matches:

            st.success(
                f"{len(matches)} transaction(s) found."
            )

            st.dataframe(
                pd.DataFrame(matches),
                use_container_width=True
            )

        else:

            st.warning(
                "No matching transaction found."
            )

    else:

        if transactions:

            st.dataframe(
                pd.DataFrame(transactions),
                use_container_width=True
            )

        else:

            st.info(
                "No transactions available yet."
            )


# ============================================================
# 9. TRENDS & MONITORING
# ============================================================

elif current_page == "Trends & Monitoring":

    st.title(
        "Trends & Monitoring",
        icon=":material/trending_up:"
    )

    st.markdown(
        "Monitor transaction volume and fraud activity "
        "over the current streaming session."
    )

    st.divider()

    transactions = st.session_state.transactions

    if transactions:

        df = pd.DataFrame(
            transactions
        )

        if "timestamp" in df.columns:

            df["timestamp"] = pd.to_datetime(
                df["timestamp"]
            )

            df = df.set_index(
                "timestamp"
            )

            st.subheader(
                "Transaction Activity"
            )

            st.line_chart(
                df.resample("1min")
                .size()
            )

        else:

            st.info(
                "Timestamp data is not yet available."
            )

    else:

        st.info(
            "Waiting for transaction stream..."
        )


# ============================================================
# 10. THRESHOLD TUNER
# ============================================================

elif current_page == "Threshold Tuner":

    st.title(
        "Threshold Tuner",
        icon=":material/tune:"
    )

    st.markdown(
        "Adjust the fraud probability threshold used "
        "to classify transactions."
    )

    st.divider()

    threshold = st.slider(
        "Fraud Classification Threshold",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01
    )

    st.metric(
        "Current Threshold",
        f"{threshold:.2f}"
    )

    st.info(
        "Lower thresholds generally increase fraud detection "
        "sensitivity, while higher thresholds generally reduce "
        "false positives. Final precision and recall should be "
        "calculated using labelled validation/test data."
    )


# ============================================================
# 11. AI ASSISTANT
# ============================================================

elif current_page == "AI Assistant":

    st.title(
        "AI Decision-Support Assistant",
        icon=":material/smart_toy:"
    )

    st.markdown(
        "Ask questions about transaction risk, model predictions "
        "and SHAP explanations."
    )

    st.divider()

    question = st.chat_input(
        "Ask about the fraud detection results..."
    )

    if question:

        with st.chat_message("user"):

            st.write(question)

        with st.chat_message("assistant"):

            st.info(
                "AI Assistant interface is ready. "
                "The next integration step will connect it "
                "to the model predictions, SHAP explanations "
                "and transaction data."
            )


# ============================================================
# 12. ABOUT
# ============================================================

elif current_page == "About":

    st.title(
        "About the XAI DSS",
        icon=":material/info:"
    )

    st.markdown("""
    ### Explainable AI Fraud Detection Decision Support System

    This system integrates:

    - Synthetic transaction generation
    - Apache Kafka for transaction streaming
    - Random Forest fraud detection
    - SHAP-based explainability
    - Real-time transaction monitoring
    - Decision-support analytics
    - Streamlit visualization

    ### Model

    The fraud detection model was trained using
    transaction features consisting of:

    - Time
    - Amount
    - V1–V28

    ### Purpose

    The system is designed to support operational
    decision-making by combining fraud prediction,
    probability estimates and explainable AI.
    """)

    st.divider()

    st.caption(
        "XAI Fraud Detection DSS"
    )