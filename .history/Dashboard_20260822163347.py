# ============================================================
# INTERPRETABLE FRAUD DETECTION & OPERATIONAL DECISION SUPPORT
# ============================================================

import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

from kafka import KafkaConsumer
from kafka.errors import KafkaError

# ============================================================
# IMAGE FILES
# ============================================================

SHIELD_ICON = "Shield Check.png"
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Detection & Operational DSS",
    page_icon=SHIELD_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 21px;
        margin-bottom: 10px;
    }
    .research-title {
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .section-title {
        text-align: center;
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .icon-center {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "transactions" not in st.session_state:
    st.session_state.transactions = []

# ============================================================
# KAFKA CONFIGURATION
# ============================================================

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "fraud-predictions"

# ============================================================
# KAFKA CONSUMER
# ============================================================

@st.cache_resource
def create_kafka_consumer():
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_SERVER,
            value_deserializer=lambda message: json.loads(message.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id="fraud-dashboard-consumer-v2",
            consumer_timeout_ms=1000,
        )
        return consumer
    except KafkaError as e:
        st.error(f"Kafka connection failed: {e}")
        return None
    except Exception as e:
        st.error(f"Kafka connection failed: {e}")
        return None

consumer = create_kafka_consumer()

# ============================================================
# READ KAFKA TRANSACTIONS
# ============================================================

def read_kafka_transactions():
    if consumer is None:
        return

    try:
        messages = consumer.poll(timeout_ms=1000, max_records=50)

        for _, records in messages.items():
            for message in records:
                transaction = message.value

                if not isinstance(transaction, dict):
                    continue

                transaction_id = transaction.get("transaction_id")
                existing_ids = {
                    t.get("transaction_id")
                    for t in st.session_state.transactions
                }

                if transaction_id is None or transaction_id not in existing_ids:
                    st.session_state.transactions.append(transaction)

    except Exception as e:
        st.warning(f"Unable to read Kafka transactions: {e}")

read_kafka_transactions()



# ============================================================
# LOAD MODEL + SHAP EXPLAINER
# Used by the Predict Fraud page
# ============================================================

import joblib
import shap
import numpy as np


@st.cache_resource
def load_prediction_model():

    model = joblib.load("rf_fraud_model.pkl")

    explainer = shap.TreeExplainer(model)

    return model, explainer


pred_model, pred_explainer = load_prediction_model()


# ============================================================
# MODEL FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.image(SHIELD_ICON, width=55)
    st.title("Fraud DSS")
    st.caption("Interpretable Fraud Detection & Operational Decision Support")
    st.caption("Developed by Adeseye Samuel Ademola")
    st.divider()

    current_page = st.radio(
        "System Navigation",
        [
            "Introduction",
            "Executive Overview",
            "Historical Model Analysis",
            "Predict Fraud",
            "Real-Time Fraud Detection",
            "Transaction Explorer",
            "Trends & Monitoring",
            "Threshold Tuner",
            "AI Decision Assistant",
            "About the Framework",
        ],
    )

    st.divider()
    st.caption("SYSTEM STATUS")

    status_col1, status_col2 = st.columns([1, 4])

    with status_col1:
        st.image(LIVE_ICON, width=30)

    with status_col2:
        if consumer is not None:
            st.success("System Online")
        else:
            st.error("Kafka Offline")

    st.caption(f"Kafka Topic: {KAFKA_TOPIC}")
    st.divider()

    if st.button("Refresh Dashboard", use_container_width=True):
        st.rerun()

# ============================================================
# 10. INTRODUCTION
# ============================================================

if current_page == "Introduction":
    st.markdown('<div class="icon-center">', unsafe_allow_html=True)
    st.image(SHIELD_ICON, width=90)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="main-title">
            INTERPRETABLE FRAUD DETECTION<br>
            & OPERATIONAL DECISION SUPPORT
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Real-Time Intelligence for Digital Banking</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="research-title">
        <b>Development of Adaptive and Interpretable Machine Learning Framework
        for Real-Time Fraud Detection and Operational Decision Support in Digital Banking</b>
        <br><br>
        Developed by <b>Adeseye Samuel Ademola</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="section-title">Background of the Study</div>', unsafe_allow_html=True)
    st.write(
        """
        The rapid growth of digital banking and electronic payment systems has increased
        the volume, velocity and complexity of financial transactions. At the same time,
        this growth has created opportunities for increasingly sophisticated fraudulent activities.

        Machine learning provides an opportunity to identify complex patterns within transaction
        data and detect potentially fraudulent behaviour. However, predictive performance alone
        is insufficient for effective operational fraud management.

        Operational teams need to understand not only whether a transaction has been classified
        as potentially fraudulent, but also why the machine learning model reached that decision.

        Interpretability is therefore important for transforming model predictions into actionable
        operational intelligence.

        This project develops an adaptive and interpretable machine learning framework that
        integrates historical fraud analysis with real-time transaction detection, explainable
        artificial intelligence and operational decision support.
        """
    )
    st.divider()

    st.markdown('<div class="section-title">Research Focus</div>', unsafe_allow_html=True)
    focus1, focus2, focus3 = st.columns(3)

    with focus1:
        st.image(MODEL_ICON, width=60)
        st.markdown("### Adaptive Machine Learning")
        st.write("Supports fraud detection within changing transaction environments and evolving fraud patterns.")

    with focus2:
        st.image(XAI_ICON, width=60)
        st.markdown("### Interpretable Machine Learning")
        st.write("Provides insight into model behaviour and identifies the factors contributing to individual predictions.")

    with focus3:
        st.image(OVERVIEW_ICON, width=60)
        st.markdown("### Operational Decision Support")
        st.write("Converts analytical outputs into information that can support operational fraud review and decision-making.")

    st.divider()

    st.markdown('<div class="section-title">Framework Workflow</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    workflow = [
        (c1, OVERVIEW_ICON, "01", "Historical Data", "Historical transaction data supports model development, evaluation and global interpretability."),
        (c2, MODEL_ICON, "02", "Machine Learning", "The Random Forest model learns patterns associated with fraudulent transactions."),
        (c3, LIVE_ICON, "03", "Real-Time Stream", "Apache Kafka transports transaction events through the real-time pipeline."),
        (c4, XAI_ICON, "04", "Explainable AI", "SHAP provides global model interpretation and local transaction-level explanations."),
        (c5, OVERVIEW_ICON, "05", "Decision Support", "Operational users receive fraud intelligence to support investigation and prioritisation."),
    ]

    for col, icon, number, title, caption in workflow:
        with col:
            st.image(icon, width=55)
            st.markdown(f"### {number}")
            st.write(title)
            st.caption(caption)

    st.divider()

    st.markdown('<div class="section-title">Model Input</div>', unsafe_allow_html=True)
    st.write(
        """
        The trained fraud detection model operates on **30 transaction features consisting of
        Time, Amount and V1-V28**.

        The V1-V28 variables represent transformed transaction characteristics in the underlying
        dataset. These same model-compatible features are maintained when generating synthetic
        transactions for the real-time Kafka stream.
        """
    )
    st.divider()

    st.markdown('<div class="section-title">Two Levels of Interpretability</div>', unsafe_allow_html=True)
    interpretation1, interpretation2 = st.columns(2)

    with interpretation1:
        st.image(XAI_ICON, width=65)
        st.markdown("### Global Interpretability")
        st.write(
            """
            Examines how the model behaves across historical transaction data.

            Examples include:
            - Global SHAP feature importance
            - Feature contribution patterns
            - Model performance
            - ROC-AUC and PR-AUC
            - Precision, recall and F1-score
            """
        )

    with interpretation2:
        st.image(TRANSACTION_ICON, width=65)
        st.markdown("### Local Interpretability")
        st.write(
            """
            Explains why the model produced a particular prediction for an individual transaction.

            Examples include:
            - Transaction fraud probability
            - Individual SHAP contributions
            - Top risk-contributing features
            - Transaction-level investigation
            """
        )

    st.divider()
    st.markdown('<div class="section-title">Purpose of the Decision Support System</div>', unsafe_allow_html=True)
    st.write(
        """
        The system is designed to support operational teams by bringing together historical model
        evaluation, real-time fraud detection, transaction-level explanations and monitoring information
        within a single interface.

        Rather than replacing human decision-making, the DSS provides evidence that can assist operational
        users in prioritising suspicious transactions, investigating risk factors and understanding
        machine learning decisions.
        """
    )
    st.divider()
    st.info("Use the navigation panel to explore historical model performance, real-time fraud detection, transaction-level explanations and operational monitoring.")

# ============================================================
# 11. EXECUTIVE OVERVIEW
# ============================================================

elif current_page == "Executive Overview":
    st.image(OVERVIEW_ICON, width=65)
    st.title("Executive Overview")
    st.write("High-level view of transaction activity, fraud risk and operational intelligence.")
    st.divider()

    transactions = st.session_state.transactions
    total = len(transactions)
    fraud = sum(t.get("prediction") == "FRAUD" for t in transactions)
    legitimate = sum(t.get("prediction") == "LEGITIMATE" for t in transactions)
    fraud_rate = fraud / total * 100 if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(TRANSACTION_ICON, width=45)
        st.metric("Transactions Processed", total)
    with c2:
        st.image(FRAUD_CAUGHT_ICON, width=45)
        st.metric("Fraud Detected", fraud)
    with c3:
        st.image(OVERVIEW_ICON, width=45)
        st.metric("Legitimate", legitimate)
    with c4:
        st.image(FRAUD_MISSED_ICON, width=45)
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    st.divider()
    st.subheader("Operational Transaction Summary")

    if total == 0:
        st.info("No transactions have been received by the dashboard yet. Start the XAI/Kafka pipeline to populate the operational dashboard.")
    else:
        st.dataframe(pd.DataFrame(transactions), use_container_width=True)

# ============================================================
# 12. HISTORICAL MODEL ANALYSIS
# ============================================================

elif current_page == "Historical Model Analysis":
    st.image(MODEL_ICON, width=65)
    st.title("Historical Model Analysis")
    st.write("Evaluation and interpretation of the trained machine learning fraud detection model.")
    st.divider()

    st.subheader("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("ROC-AUC", "0.961")
    with c2:
        st.metric("PR-AUC", "0.813")
    with c3:
        st.metric("Fraud Precision", "0.90")
    with c4:
        st.metric("Fraud Recall", "0.77")

    st.divider()
    st.subheader("Global Model Interpretability")
    st.image(XAI_ICON, width=60)
    st.info("Global SHAP analysis identifies the features that have the greatest influence on model predictions across historical transaction data.")

    if model_error:
        st.warning(f"Model/SHAP files could not be loaded: {model_error}")
    else:
        st.success("Random Forest model and SHAP explainer loaded successfully.")

# ============================================================
# 13. PREDICT FRAUD
# ============================================================

elif current_page == "Predict Fraud":
    st.image(XAI_ICON, width=65)
    st.title("Predict Fraud")
    st.write(
        """
        Submit a transaction manually, or upload a CSV of transactions, to check the model's
        fraud prediction and see the top contributing factors behind each result.
        """
    )
    st.divider()

    if pred_model is None or pred_explainer is None:
        st.error("The prediction model could not be loaded. Make sure rf_fraud_model.pkl is in the same folder as Dashboard.py.")
    else:
        tab_manual, tab_batch = st.tabs(["Manual Entry", "Batch CSV Upload"])

        def predict_and_explain(input_df: pd.DataFrame):
            prediction = pred_model.predict(input_df)[0]
            probability = pred_model.predict_proba(input_df)[0][1]
            result = "FRAUD" if prediction == 1 else "LEGITIMATE"

            shap_values = pred_explainer.shap_values(input_df)

            if isinstance(shap_values, list):
                fraud_shap = np.asarray(shap_values[1])[0]
            else:
                shap_values = np.asarray(shap_values)
                if shap_values.ndim == 3:
                    fraud_shap = shap_values[0, :, 1]
                elif shap_values.ndim == 2:
                    fraud_shap = shap_values[0]
                else:
                    fraud_shap = shap_values

            shap_df = pd.DataFrame(
                {
                    "Feature": FEATURE_COLUMNS,
                    "Feature_Value": input_df.iloc[0].values,
                    "SHAP_Value": fraud_shap,
                }
            )
            shap_df["Absolute_SHAP"] = shap_df["SHAP_Value"].abs()
            shap_df = shap_df.sort_values("Absolute_SHAP", ascending=False)
            return result, probability, shap_df

        with tab_manual:
            st.subheader("Enter Transaction Details")
            st.caption(
                "V1-V28 are anonymised PCA-transformed features. If you do not have values, leave them at 0.0 for testing."
            )

            with st.form("manual_prediction_form"):
                col_a, col_b = st.columns(2)

                with col_a:
                    input_time = st.number_input("Time", value=0.0, format="%.6f")

                with col_b:
                    input_amount = st.number_input("Amount", value=0.0, format="%.6f")

                st.markdown("**V1 - V28 (PCA features)**")
                v_inputs = {}
                v_cols_ui = st.columns(4)

                for i in range(1, 29):
                    col = v_cols_ui[(i - 1) % 4]
                    with col:
                        v_inputs[f"V{i}"] = st.number_input(
                            f"V{i}", value=0.0, format="%.6f", key=f"v_{i}"
                        )

                submitted = st.form_submit_button("Predict Fraud", use_container_width=True)

            if submitted:
                row = {"Time": input_time}
                row.update(v_inputs)
                row["Amount"] = input_amount
                input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

                result, probability, shap_df = predict_and_explain(input_df)
                st.divider()

                if result == "FRAUD":
                    st.error(f"FRAUD DETECTED - Probability: {probability:.2%}")
                else:
                    st.success(f"LEGITIMATE - Fraud Probability: {probability:.2%}")

                st.subheader("Top Contributing Features")
                top5 = shap_df.head(5).copy()
                top5["Direction"] = np.where(
                    top5["SHAP_Value"] >= 0,
                    "Pushes toward Fraud",
                    "Pushes toward Legitimate",
                )
                st.dataframe(
                    top5[["Feature", "Feature_Value", "SHAP_Value", "Direction"]],
                    use_container_width=True,
                    hide_index=True,
                )

        with tab_batch:
            st.subheader("Upload a CSV of Transactions")
            st.caption("The CSV must contain exactly these model features: Time, V1-V28 and Amount.")

            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_predict_upload")

            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                missing_cols = set(FEATURE_COLUMNS) - set(batch_df.columns)

                if missing_cols:
                    st.error(f"Uploaded file is missing required columns: {sorted(missing_cols)}")
                else:
                    X_batch = batch_df[FEATURE_COLUMNS]
                    predictions = pred_model.predict(X_batch)
                    probabilities = pred_model.predict_proba(X_batch)[:, 1]

                    results_df = batch_df.copy()
                    results_df["prediction"] = np.where(predictions == 1, "FRAUD", "LEGITIMATE")
                    results_df["fraud_probability"] = probabilities

                    total_batch = len(results_df)
                    fraud_batch = int((predictions == 1).sum())

                    bc1, bc2, bc3 = st.columns(3)
                    with bc1:
                        st.metric("Transactions Uploaded", total_batch)
                    with bc2:
                        st.metric("Flagged as Fraud", fraud_batch)
                    with bc3:
                        rate = fraud_batch / total_batch * 100 if total_batch else 0
                        st.metric("Fraud Rate", f"{rate:.2f}%")

                    st.divider()
                    st.subheader("Flagged Transactions")
                    fraud_results = results_df[results_df["prediction"] == "FRAUD"].sort_values(
                        "fraud_probability", ascending=False
                    )

                    if fraud_results.empty:
                        st.info("No transactions in this batch were flagged as fraud.")
                    else:
                        st.dataframe(fraud_results, use_container_width=True)

                    st.divider()
                    st.subheader("Full Results")
                    st.dataframe(results_df, use_container_width=True)

                    csv_download = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results as CSV",
                        data=csv_download,
                        file_name="fraud_predictions_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

# ============================================================
# 14. REAL-TIME FRAUD DETECTION
# ============================================================

elif current_page == "Real-Time Fraud Detection":
    st.image(LIVE_ICON, width=65)
    st.title("Real-Time Fraud Detection")
    st.write("Live transaction monitoring and fraud classification from the Kafka transaction stream.")
    st.divider()

    transactions = st.session_state.transactions

    if not transactions:
        st.info("Waiting for transactions from Kafka...")
    else:
        latest = transactions[-1]
        prediction = latest.get("prediction", "UNKNOWN")
        probability = latest.get("fraud_probability", 0)

        try:
            probability = float(probability)
            if probability > 1:
                probability /= 100
        except (TypeError, ValueError):
            probability = 0

        if prediction == "FRAUD":
            st.error(f"FRAUD DETECTED - Probability: {probability:.2%}")
        elif prediction == "LEGITIMATE":
            st.success(f"LEGITIMATE TRANSACTION - Fraud Probability: {probability:.2%}")
        else:
            st.warning("Transaction classification unavailable.")

        st.divider()
        st.subheader("Latest Transaction")
        st.json(latest)

        if latest.get("top_shap_features"):
            st.subheader("Top SHAP Risk Factors")
            st.dataframe(pd.DataFrame(latest["top_shap_features"]), use_container_width=True, hide_index=True)

# ============================================================
# 15. TRANSACTION EXPLORER
# ============================================================

elif current_page == "Transaction Explorer":
    st.image(TRANSACTION_ICON, width=65)
    st.title("Transaction Explorer")
    st.write("Investigate individual transactions, model predictions and transaction-level risk information.")
    st.divider()

    query = st.text_input("Transaction ID", placeholder="Enter transaction ID...")
    transactions = st.session_state.transactions

    if query:
        matches = [
            t for t in transactions
            if query.lower() in str(t.get("transaction_id", "")).lower()
        ]

        if matches:
            st.success(f"{len(matches)} transaction(s) found.")
            st.dataframe(pd.DataFrame(matches), use_container_width=True)
        else:
            st.warning("No matching transaction found.")
    elif transactions:
        st.dataframe(pd.DataFrame(transactions), use_container_width=True)
    else:
        st.info("No transaction data available.")

# ============================================================
# 16. TRENDS & MONITORING
# ============================================================

elif current_page == "Trends & Monitoring":
    st.image(TREND_ICON, width=65)
    st.title("Trends & Monitoring")
    st.write("Monitor transaction activity and fraud patterns across the real-time transaction stream.")
    st.divider()

    transactions = st.session_state.transactions

    if transactions:
        df = pd.DataFrame(transactions)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

            if not df.empty:
                chart = df.set_index("timestamp").resample("1min").size()
                st.subheader("Transaction Volume")
                st.line_chart(chart)
            else:
                st.info("Valid timestamp information is not available.")
        else:
            st.info("Timestamp information is not available yet.")
    else:
        st.info("Waiting for transaction data...")

# ============================================================
# 17. THRESHOLD TUNER
# ============================================================

elif current_page == "Threshold Tuner":
    st.image(THRESHOLD_ICON, width=65)
    st.title("Fraud Detection Threshold")
    st.write("Explore the relationship between fraud probability thresholds and operational classification decisions.")
    st.divider()

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
    )

    st.metric("Selected Threshold", f"{threshold:.2f}")
    st.divider()
    st.info(
        """
        A lower threshold increases sensitivity to potentially fraudulent transactions but may increase false positives.

        A higher threshold may reduce false alarms but can increase the risk of missed fraud.

        The final operational threshold should be selected using validation data and the relative cost of false positives and false negatives.
        """
    )

# ============================================================
# 18. AI DECISION ASSISTANT
# ============================================================

elif current_page == "AI Decision Assistant":
    st.image(AI_ASSISTANT_ICON, width=65)
    st.title("AI Assistant")
    st.write("An intelligent interface for assisting operational users in interpreting fraud detection results, transaction risk and model explanations.")
    st.divider()

    question = st.chat_input("Ask about a transaction, prediction or model explanation...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.info(
                "The AI Decision Assistant interface is ready. The next integration stage will connect the assistant to transaction predictions, SHAP explanations and operational data."
            )

# ============================================================
# 19. ABOUT THE FRAMEWORK
# ============================================================

elif current_page == "About the Framework":
    st.image(INFO_ICON, width=65)
    st.title("About the Framework")

    st.markdown(
        """
        ### Development of Adaptive and Interpretable Machine Learning Framework for Real-Time Fraud Detection and Operational Decision Support in Digital Banking

        **Developed by: Adeseye Samuel Ademola**
        """
    )
    st.divider()

    st.write(
        """
        The framework integrates machine learning, real-time transaction streaming,
        Explainable Artificial Intelligence and operational decision-support capabilities.
        """
    )

    st.subheader("Core Technologies")
    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Pandas / NumPy",
                "Scikit-learn",
                "Random Forest",
                "Apache Kafka",
                "SHAP",
                "Streamlit",
            ],
            "Purpose": [
                "Application and ML development",
                "Data processing",
                "Machine learning",
                "Fraud classification",
                "Real-time transaction streaming",
                "Model explainability",
                "Decision-support interface",
            ],
        }
    )
    st.dataframe(technologies, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Model Input")
    st.write(
        """
        The trained fraud detection model uses 30 features:

        **Time, V1-V28 and Amount.**
        """
    )

    st.divider()
    st.subheader("Interpretability")
    st.write(
        """
        The framework addresses interpretability at two levels:

        **Global interpretability:** Understanding the behaviour of the model across historical transaction data.

        **Local interpretability:** Understanding why the model produced a particular prediction for an individual transaction.

        SHAP is used to provide feature-level explanations of model predictions.
        """
    )

    st.divider()
    st.subheader("Operational Decision Support")
    st.write(
        """
        The DSS does not replace human decision-making. Instead, it provides operational users
        with predictive and explanatory information that can support transaction investigation,
        prioritization and fraud-response decisions.
        """
    )

    st.divider()
    st.caption("Interpretable Fraud Detection & Operational Decision Support System")
    st.caption("Developed by Adeseye Samuel Ademola")