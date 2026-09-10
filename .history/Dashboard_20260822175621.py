# ============================================================
# INTERPRETABLE FRAUD DETECTION & OPERATIONAL DECISION SUPPORT
# COMPLETE DASHBOARD
# ============================================================

import os
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

from lime.lime_tabular import LimeTabularExplainer

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

if "feedback_records" not in st.session_state:
    st.session_state.feedback_records = []

if "adaptation_history" not in st.session_state:
    st.session_state.adaptation_history = []


# ============================================================
# MODEL FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ============================================================
# KAFKA CONFIGURATION
# ============================================================

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "fraud-predictions"


# ============================================================
# ADAPTIVE MONITORING CONFIGURATION
# ============================================================

ADAPTATION_WINDOW = 100

MIN_TRANSACTIONS_FOR_ADAPTATION = 30

FRAUD_RATE_CHANGE_THRESHOLD = 2.0

MIN_FEEDBACK_FOR_ADAPTATION = 10


# ============================================================
# COST / VALUE CONFIGURATION
# ============================================================

DEFAULT_AVERAGE_FRAUD_LOSS = 100000.0

DEFAULT_FALSE_ALARM_COST = 500.0

DEFAULT_INVESTIGATION_COST = 1000.0


# ============================================================
# KAFKA CONSUMER
# ============================================================

@st.cache_resource
def create_kafka_consumer():

    try:

        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_SERVER,
            value_deserializer=lambda message:
                json.loads(
                    message.decode("utf-8")
                ),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id="fraud-dashboard-consumer-v2",
            consumer_timeout_ms=1000,
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
# READ KAFKA TRANSACTIONS
# ============================================================

def read_kafka_transactions():

    if consumer is None:
        return

    try:

        messages = consumer.poll(
            timeout_ms=1000,
            max_records=50
        )

        for _, records in messages.items():

            for message in records:

                transaction = message.value

                if not isinstance(
                    transaction,
                    dict
                ):
                    continue

                transaction_id = (
                    transaction.get(
                        "transaction_id"
                    )
                )

                existing_ids = {
                    t.get(
                        "transaction_id"
                    )
                    for t in st.session_state.transactions
                }

                if (
                    transaction_id is None
                    or transaction_id not in existing_ids
                ):

                    st.session_state.transactions.append(
                        transaction
                    )

    except Exception as e:

        st.warning(
            f"Unable to read Kafka transactions: {e}"
        )


read_kafka_transactions()


# ============================================================
# LOAD HISTORICAL MODEL RESULTS
# ============================================================

@st.cache_data
def load_historical_results():

    summary = None
    predictions = None
    shap_importance = None

    # --------------------------------------------------------
    # HISTORICAL SUMMARY
    # --------------------------------------------------------

    if os.path.exists(
        "historical_summary.json"
    ):

        try:

            with open(
                "historical_summary.json",
                "r"
            ) as f:

                summary = json.load(f)

        except Exception as e:

            st.warning(
                f"Unable to load historical_summary.json: {e}"
            )


    # --------------------------------------------------------
    # HISTORICAL PREDICTIONS
    # --------------------------------------------------------

    if os.path.exists(
        "historical_predictions.csv"
    ):

        try:

            predictions = pd.read_csv(
                "historical_predictions.csv"
            )

        except Exception as e:

            st.warning(
                f"Unable to load historical_predictions.csv: {e}"
            )


    # --------------------------------------------------------
    # GLOBAL SHAP
    # --------------------------------------------------------

    if os.path.exists(
        "historical_global_shap.csv"
    ):

        try:

            shap_importance = pd.read_csv(
                "historical_global_shap.csv"
            )

        except Exception as e:

            st.warning(
                f"Unable to load historical_global_shap.csv: {e}"
            )

    return (
        summary,
        predictions,
        shap_importance
    )


(
    historical_summary,
    historical_predictions,
    historical_shap
) = load_historical_results()


# ============================================================
# LOAD MODEL + SHAP + LIME
# ============================================================

@st.cache_resource
def load_prediction_model():

    model = joblib.load(
        "rf_fraud_model.pkl"
    )

    shap_explainer = shap.TreeExplainer(
        model
    )

    historical_data = pd.read_csv(
        "historical_test_data.csv"
    )

    background_data = historical_data[
        FEATURE_COLUMNS
    ]

    lime_explainer = LimeTabularExplainer(
        training_data=background_data.values,

        feature_names=FEATURE_COLUMNS,

        class_names=[
            "LEGITIMATE",
            "FRAUD"
        ],

        mode="classification",

        discretize_continuous=True,

        random_state=42
    )

    return (
        model,
        shap_explainer,
        lime_explainer
    )


# ============================================================
# INITIALISE MODEL
# ============================================================

pred_model = None
pred_explainer = None
lime_explainer = None
model_error = None


try:

    (
        pred_model,
        pred_explainer,
        lime_explainer
    ) = load_prediction_model()

except Exception as e:

    model_error = str(e)


# ============================================================
# PREDICT + EXPLAIN FUNCTION
# SHAP + LIME
# ============================================================

def predict_and_explain(
    input_df: pd.DataFrame
):

    if pred_model is None:

        raise RuntimeError(
            "Prediction model is not available."
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = pred_model.predict(
        input_df
    )[0]

    probability = pred_model.predict_proba(
        input_df
    )[0][1]

    result = (
        "FRAUD"
        if prediction == 1
        else "LEGITIMATE"
    )


    # ========================================================
    # SHAP
    # ========================================================

    shap_values = pred_explainer.shap_values(
        input_df
    )


    if isinstance(
        shap_values,
        list
    ):

        fraud_shap = np.asarray(
            shap_values[1]
        )[0]

    else:

        shap_values = np.asarray(
            shap_values
        )

        if shap_values.ndim == 3:

            fraud_shap = shap_values[
                0,
                :,
                1
            ]

        elif shap_values.ndim == 2:

            fraud_shap = shap_values[
                0
            ]

        else:

            fraud_shap = shap_values


    # ========================================================
    # SHAP DATAFRAME
    # ========================================================

    shap_df = pd.DataFrame(
        {
            "Feature":
                FEATURE_COLUMNS,

            "Feature_Value":
                input_df.iloc[0].values,

            "SHAP_Value":
                fraud_shap
        }
    )


    shap_df[
        "Absolute_SHAP"
    ] = shap_df[
        "SHAP_Value"
    ].abs()


    shap_df = shap_df.sort_values(
        "Absolute_SHAP",
        ascending=False
    )


    # ========================================================
    # LIME
    # ========================================================

    lime_explanation = (
        lime_explainer.explain_instance(
            input_df.iloc[0].values,

            pred_model.predict_proba,

            num_features=10
        )
    )


    lime_df = pd.DataFrame(
        lime_explanation.as_list(),

        columns=[
            "Feature",
            "LIME_Weight"
        ]
    )


    lime_df[
        "Absolute_LIME"
    ] = lime_df[
        "LIME_Weight"
    ].abs()


    lime_df = lime_df.sort_values(
        "Absolute_LIME",
        ascending=False
    )


    return (
        result,
        probability,
        shap_df,
        lime_df
    )


# ============================================================
# HISTORICAL BASELINE FRAUD RATE
# ============================================================

def get_historical_fraud_rate():

    if historical_summary is None:

        return 0.0

    try:

        total = float(
            historical_summary.get(
                "total_transactions",
                0
            )
        )

        fraud = float(
            historical_summary.get(
                "actual_fraud_count",
                0
            )
        )

        if total <= 0:
            return 0.0

        return fraud / total

    except Exception:

        return 0.0


# ============================================================
# ADAPTIVE MONITORING
# ============================================================

def assess_adaptation_status(
    transactions
):

    if not transactions:

        return {
            "status":
                "INSUFFICIENT DATA",

            "recent_transactions":
                0,

            "recent_fraud_rate":
                0.0,

            "historical_fraud_rate":
                get_historical_fraud_rate(),

            "change_ratio":
                0.0,

            "reason":
                "No real-time transactions available."
        }


    recent_transactions = transactions[
        -ADAPTATION_WINDOW:
    ]


    transaction_count = len(
        recent_transactions
    )


    if (
        transaction_count
        <
        MIN_TRANSACTIONS_FOR_ADAPTATION
    ):

        return {
            "status":
                "MONITORING",

            "recent_transactions":
                transaction_count,

            "recent_fraud_rate":
                0.0,

            "historical_fraud_rate":
                get_historical_fraud_rate(),

            "change_ratio":
                0.0,

            "reason":
                (
                    f"Only {transaction_count} recent "
                    "transactions are available. "
                    f"At least "
                    f"{MIN_TRANSACTIONS_FOR_ADAPTATION} "
                    "are required."
                )
        }


    fraud_count = sum(
        1
        for transaction
        in recent_transactions

        if transaction.get(
            "prediction"
        ) == "FRAUD"
    )


    recent_fraud_rate = (
        fraud_count /
        transaction_count
    )


    historical_fraud_rate = (
        get_historical_fraud_rate()
    )


    if historical_fraud_rate <= 0:

        change_ratio = (
            float("inf")
            if recent_fraud_rate > 0
            else 0.0
        )

    else:

        change_ratio = (
            recent_fraud_rate /
            historical_fraud_rate
        )


    if (
        change_ratio
        >=
        FRAUD_RATE_CHANGE_THRESHOLD
    ):

        status = (
            "ADAPTATION REQUIRED"
        )

        reason = (
            "Recent fraud behaviour is substantially "
            "different from the historical baseline. "
            "The model should be reassessed using recent "
            "transaction and feedback data."
        )

    else:

        status = "STABLE"

        reason = (
            "Recent fraud behaviour remains within "
            "the defined monitoring range."
        )


    return {

        "status":
            status,

        "recent_transactions":
            transaction_count,

        "recent_fraud_rate":
            recent_fraud_rate,

        "historical_fraud_rate":
            historical_fraud_rate,

        "change_ratio":
            change_ratio,

        "reason":
            reason
    }


# ============================================================
# SAVE OPERATIONAL FEEDBACK
# ============================================================

def save_feedback(
    transaction,
    feedback_label,
    reviewer_comment=""
):

    feedback_record = {
        "feedback_id":
            str(uuid.uuid4()),

        "timestamp":
            datetime.now().isoformat(),

        "transaction_id":
            transaction.get(
                "transaction_id",
                ""
            ),

        "model_prediction":
            transaction.get(
                "prediction",
                ""
            ),

        "fraud_probability":
            transaction.get(
                "fraud_probability",
                0
            ),

        "operational_label":
            feedback_label,

        "reviewer_comment":
            reviewer_comment
    }


    # Add model features where available

    for feature in FEATURE_COLUMNS:

        feedback_record[
            feature
        ] = transaction.get(
            feature,
            np.nan
        )


    st.session_state.feedback_records.append(
        feedback_record
    )


    feedback_df = pd.DataFrame(
        st.session_state.feedback_records
    )


    feedback_df.to_csv(
        "operational_feedback.csv",
        index=False
    )


    return feedback_record


# ============================================================
# LOAD EXISTING FEEDBACK
# ============================================================

def load_existing_feedback():

    if os.path.exists(
        "operational_feedback.csv"
    ):

        try:

            return pd.read_csv(
                "operational_feedback.csv"
            ).to_dict(
                orient="records"
            )

        except Exception:

            return []

    return []


if not st.session_state.feedback_records:

    st.session_state.feedback_records = (
        load_existing_feedback()
    )


# ============================================================
# CREATE ADAPTATION DATASET
# ============================================================

def create_adaptation_dataset():

    if not st.session_state.feedback_records:

        return None


    feedback_df = pd.DataFrame(
        st.session_state.feedback_records
    )


    available_features = [
        col
        for col in FEATURE_COLUMNS
        if col in feedback_df.columns
    ]


    if not available_features:

        return None


    adaptation_df = feedback_df[
        available_features
        +
        [
            "operational_label"
        ]
    ].copy()


    # Only feedback with known operational outcomes

    adaptation_df = adaptation_df[
        adaptation_df[
            "operational_label"
        ].notna()
    ]


    # Convert operational labels to model target

    adaptation_df[
        "actual_label"
    ] = np.where(
        adaptation_df[
            "operational_label"
        ].astype(str).str.upper()
        == "CONFIRMED FRAUD",
        1,
        0
    )


    adaptation_df.to_csv(
        "adaptation_dataset.csv",
        index=False
    )


    return adaptation_df


# ============================================================
# CREATE ADAPTATION REQUEST
# ============================================================

def create_adaptation_request(
    adaptation_status
):

    adaptation_dataset = (
        create_adaptation_dataset()
    )


    feedback_count = len(
        st.session_state.feedback_records
    )


    request = {

        "request_id":
            str(uuid.uuid4()),

        "created_at":
            datetime.now().isoformat(),

        "status":
            "ADAPTATION REQUIRED",

        "reason":
            adaptation_status.get(
                "reason",
                ""
            ),

        "recent_transactions":
            adaptation_status.get(
                "recent_transactions",
                0
            ),

        "recent_fraud_rate":
            adaptation_status.get(
                "recent_fraud_rate",
                0
            ),

        "historical_fraud_rate":
            adaptation_status.get(
                "historical_fraud_rate",
                0
            ),

        "behaviour_change_ratio":
            adaptation_status.get(
                "change_ratio",
                0
            ),

        "feedback_records":
            feedback_count,

        "minimum_feedback_required":
            MIN_FEEDBACK_FOR_ADAPTATION,

        "adaptation_dataset":
            (
                "adaptation_dataset.csv"
                if adaptation_dataset is not None
                else None
            ),

        "recommended_action":
            (
                "Review recent transaction behaviour, "
                "validate operational feedback, retrain "
                "or recalibrate the model using approved "
                "new labelled data, validate the candidate "
                "model and deploy only after approval."
            )
    }


    with open(
        "adaptation_request.json",
        "w"
    ) as f:

        json.dump(
            request,
            f,
            indent=4
        )


    return request


# ============================================================
# COST / VALUE CALCULATION
# ============================================================

def calculate_cost_value(
    transactions,
    average_fraud_loss,
    false_alarm_cost,
    investigation_cost
):

    if not transactions:

        return {
            "fraud_caught": 0,
            "fraud_missed": 0,
            "false_alarms": 0,
            "estimated_value_protected": 0,
            "estimated_false_alarm_cost": 0,
            "estimated_investigation_cost": 0,
            "estimated_net_value": 0
        }


    fraud_caught = 0
    fraud_missed = 0
    false_alarms = 0


    for transaction in transactions:

        prediction = transaction.get(
            "prediction"
        )

        actual = transaction.get(
            "actual_label"
        )


        # Actual label can be numeric

        try:

            actual = int(actual)

        except (
            TypeError,
            ValueError
        ):

            actual = None


        if actual == 1:

            if prediction == "FRAUD":

                fraud_caught += 1

            elif prediction == "LEGITIMATE":

                fraud_missed += 1


        elif actual == 0:

            if prediction == "FRAUD":

                false_alarms += 1


    estimated_value_protected = (
        fraud_caught *
        average_fraud_loss
    )


    estimated_false_alarm_cost = (
        false_alarms *
        false_alarm_cost
    )


    estimated_investigation_cost = (
        fraud_caught *
        investigation_cost
    )


    estimated_net_value = (
        estimated_value_protected
        -
        estimated_false_alarm_cost
        -
        estimated_investigation_cost
    )


    return {

        "fraud_caught":
            fraud_caught,

        "fraud_missed":
            fraud_missed,

        "false_alarms":
            false_alarms,

        "estimated_value_protected":
            estimated_value_protected,

        "estimated_false_alarm_cost":
            estimated_false_alarm_cost,

        "estimated_investigation_cost":
            estimated_investigation_cost,

        "estimated_net_value":
            estimated_net_value
    }


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.image(
        SHIELD_ICON,
        width=55
    )

    st.title(
        "Fraud DSS"
    )

    st.caption(
        "Interpretable Fraud Detection & "
        "Operational Decision Support"
    )

    st.caption(
        "Developed by Adeseye Samuel Ademola"
    )

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
            "Operational Feedback",
            "Adaptive Monitoring",
            "Cost & Value Analysis",
            "AI Decision Assistant",
            "Framework Validation",
            "About the Framework",
        ],
    )


    st.divider()

    st.caption(
        "SYSTEM STATUS"
    )


    status_col1, status_col2 = st.columns(
        [1, 4]
    )


    with status_col1:

        st.image(
            LIVE_ICON,
            width=30
        )


    with status_col2:

        if consumer is not None:

            st.success(
                "System Online"
            )

        else:

            st.error(
                "Kafka Offline"
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


# ============================================================
# 10. INTRODUCTION
# ============================================================

if current_page == "Introduction":

    st.markdown(
        '<div class="icon-center">',
        unsafe_allow_html=True
    )

    st.image(
        SHIELD_ICON,
        width=90
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


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
        '<div class="subtitle">'
        'Real-Time Intelligence for Digital Banking'
        '</div>',
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


    st.markdown(
        '<div class="section-title">'
        'Background of the Study'
        '</div>',
        unsafe_allow_html=True
    )


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
        artificial intelligence, operational feedback, adaptive monitoring and decision support.
        """
    )


    st.divider()


    st.markdown(
        '<div class="section-title">'
        'Research Focus'
        '</div>',
        unsafe_allow_html=True
    )


    focus1, focus2, focus3 = st.columns(3)


    with focus1:

        st.image(
            MODEL_ICON,
            width=60
        )

        st.markdown(
            "### Adaptive Machine Learning"
        )

        st.write(
            "Supports fraud detection within changing transaction environments and evolving fraud patterns."
        )


    with focus2:

        st.image(
            XAI_ICON,
            width=60
        )

        st.markdown(
            "### Interpretable Machine Learning"
        )

        st.write(
            "Provides insight into model behaviour and identifies the factors contributing to individual predictions."
        )


    with focus3:

        st.image(
            OVERVIEW_ICON,
            width=60
        )

        st.markdown(
            "### Operational Decision Support"
        )

        st.write(
            "Converts analytical outputs into information that can support operational fraud review and decision-making."
        )


    st.divider()


    st.markdown(
        '<div class="section-title">'
        'Framework Workflow'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4, c5 = st.columns(5)


    workflow = [

        (
            c1,
            OVERVIEW_ICON,
            "01",
            "Historical Data",
            "Historical transaction data supports model development, evaluation and global interpretability."
        ),

        (
            c2,
            MODEL_ICON,
            "02",
            "Machine Learning",
            "The Random Forest model learns patterns associated with fraudulent transactions."
        ),

        (
            c3,
            LIVE_ICON,
            "03",
            "Real-Time Stream",
            "Apache Kafka transports transaction events through the real-time pipeline."
        ),

        (
            c4,
            XAI_ICON,
            "04",
            "Explainable AI",
            "SHAP and LIME provide global and local explanations."
        ),

        (
            c5,
            OVERVIEW_ICON,
            "05",
            "Decision Support",
            "Operational users receive fraud intelligence and feedback mechanisms."
        ),
    ]


    for (
        col,
        icon,
        number,
        title,
        caption
    ) in workflow:

        with col:

            st.image(
                icon,
                width=55
            )

            st.markdown(
                f"### {number}"
            )

            st.write(
                title
            )

            st.caption(
                caption
            )


    st.divider()


    st.markdown(
        '<div class="section-title">'
        'Model Input'
        '</div>',
        unsafe_allow_html=True
    )


    st.write(
        """
        The trained fraud detection model operates on 30 transaction features consisting of
        **Time, Amount and V1-V28**.
        """
    )


# ============================================================
# 11. EXECUTIVE OVERVIEW
# ============================================================

elif current_page == "Executive Overview":

    st.image(
        OVERVIEW_ICON,
        width=65
    )

    st.title(
        "Executive Overview"
    )

    st.write(
        "High-level view of transaction activity, fraud risk and operational intelligence."
    )

    st.divider()


    transactions = (
        st.session_state.transactions
    )


    total = len(
        transactions
    )


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


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Transactions Processed",
            total
        )


    with c2:

        st.metric(
            "Fraud Detected",
            fraud
        )


    with c3:

        st.metric(
            "Legitimate",
            legitimate
        )


    with c4:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )


    st.divider()


    st.subheader(
        "Operational Transaction Summary"
    )


    if total == 0:

        st.info(
            "No transactions have been received by the dashboard yet."
        )

    else:

        st.dataframe(
            pd.DataFrame(
                transactions
            ),
            use_container_width=True
        )


# ============================================================
# 12. HISTORICAL MODEL ANALYSIS
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
        "Evaluation and interpretation of the trained Random Forest model."
    )

    st.divider()


    if historical_summary is None:

        st.error(
            "Historical analysis files could not be loaded."
        )

    else:

        total_transactions = int(
            historical_summary.get(
                "total_transactions",
                0
            )
        )

        actual_fraud_count = int(
            historical_summary.get(
                "actual_fraud_count",
                0
            )
        )

        fraud_caught = int(
            historical_summary.get(
                "fraud_caught",
                0
            )
        )

        fraud_missed = int(
            historical_summary.get(
                "fraud_missed",
                0
            )
        )

        false_alarms = int(
            historical_summary.get(
                "false_alarms",
                0
            )
        )

        detection_rate = float(
            historical_summary.get(
                "detection_rate",
                0
            )
        )

        roc_auc = float(
            historical_summary.get(
                "roc_auc",
                0
            )
        )

        pr_auc = float(
            historical_summary.get(
                "pr_auc",
                0
            )
        )

        fraud_precision = float(
            historical_summary.get(
                "fraud_precision",
                0
            )
        )

        fraud_recall = float(
            historical_summary.get(
                "fraud_recall",
                0
            )
        )

        fraud_f1 = float(
            historical_summary.get(
                "fraud_f1",
                0
            )
        )


        st.subheader(
            "Model Performance"
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "ROC-AUC",
                f"{roc_auc:.3f}"
            )


        with c2:

            st.metric(
                "PR-AUC",
                f"{pr_auc:.3f}"
            )


        with c3:

            st.metric(
                "Fraud Precision",
                f"{fraud_precision:.2f}"
            )


        with c4:

            st.metric(
                "Fraud Recall",
                f"{fraud_recall:.2f}"
            )


        st.divider()


        st.subheader(
            "Historical Test Dataset"
        )


        d1, d2, d3, d4 = st.columns(4)


        with d1:

            st.metric(
                "Transactions Evaluated",
                f"{total_transactions:,}"
            )


        with d2:

            st.metric(
                "Actual Fraud",
                f"{actual_fraud_count:,}"
            )


        with d3:

            st.metric(
                "Fraud Detected",
                f"{fraud_caught:,}"
            )


        with d4:

            st.metric(
                "Fraud Missed",
                f"{fraud_missed:,}"
            )


        st.divider()


        st.subheader(
            "Operational Fraud Detection Performance"
        )


        o1, o2, o3 = st.columns(3)


        with o1:

            st.metric(
                "Detection Rate",
                f"{detection_rate:.2%}"
            )


        with o2:

            st.metric(
                "False Alarms",
                f"{false_alarms:,}"
            )


        with o3:

            st.metric(
                "Fraud F1",
                f"{fraud_f1:.3f}"
            )


        st.divider()


        st.subheader(
            "Global Model Interpretability"
        )


        if historical_shap is not None:

            st.dataframe(
                historical_shap.head(10),
                use_container_width=True,
                hide_index=True
            )


            chart = (
                historical_shap
                .head(10)
                .sort_values(
                    "Mean_Abs_SHAP"
                )
                .set_index(
                    "Feature"
                )
            )


            st.bar_chart(
                chart[
                    "Mean_Abs_SHAP"
                ]
            )


        st.divider()


        if historical_predictions is not None:

            st.subheader(
                "Historical Prediction Results"
            )

            display_columns = [
                "prediction",
                "fraud_probability",
                "actual_label"
            ]

            available_columns = [
                col
                for col in display_columns
                if col in historical_predictions.columns
            ]

            st.dataframe(
                historical_predictions[
                    available_columns
                ],
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 13. PREDICT FRAUD
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
        "Predict a transaction and obtain SHAP and LIME explanations."
    )

    st.divider()


    if model_error:

        st.error(
            f"Model could not be loaded: {model_error}"
        )

    elif pred_model is None:

        st.error(
            "Prediction model is unavailable."
        )

    else:

        tab_manual, tab_batch = st.tabs(
            [
                "Manual Entry",
                "Batch CSV Upload"
            ]
        )


        # ====================================================
        # MANUAL ENTRY
        # ====================================================

        with tab_manual:

            st.subheader(
                "Enter Transaction Details"
            )


            with st.form(
                "manual_prediction_form"
            ):

                col_a, col_b = st.columns(2)


                with col_a:

                    input_time = st.number_input(
                        "Time",
                        value=0.0,
                        format="%.6f"
                    )


                with col_b:

                    input_amount = st.number_input(
                        "Amount",
                        value=0.0,
                        format="%.6f"
                    )


                st.markdown(
                    "**V1 - V28 (PCA features)**"
                )


                v_inputs = {}

                v_cols_ui = st.columns(4)


                for i in range(
                    1,
                    29
                ):

                    col = v_cols_ui[
                        (i - 1) % 4
                    ]


                    with col:

                        v_inputs[
                            f"V{i}"
                        ] = st.number_input(
                            f"V{i}",
                            value=0.0,
                            format="%.6f",
                            key=f"manual_v_{i}"
                        )


                submitted = st.form_submit_button(
                    "Predict Fraud",
                    use_container_width=True
                )


            if submitted:

                row = {
                    "Time":
                        input_time
                }


                row.update(
                    v_inputs
                )


                row[
                    "Amount"
                ] = input_amount


                input_df = pd.DataFrame(
                    [row],
                    columns=FEATURE_COLUMNS
                )


                (
                    result,
                    probability,
                    shap_df,
                    lime_df
                ) = predict_and_explain(
                    input_df
                )


                st.divider()


                if result == "FRAUD":

                    st.error(
                        f"FRAUD DETECTED - "
                        f"Probability: {probability:.2%}"
                    )

                else:

                    st.success(
                        f"LEGITIMATE - "
                        f"Fraud Probability: {probability:.2%}"
                    )


                # SHAP

                st.subheader(
                    "SHAP Explanation"
                )


                top_shap = (
                    shap_df.head(10).copy()
                )


                top_shap[
                    "Direction"
                ] = np.where(
                    top_shap[
                        "SHAP_Value"
                    ] >= 0,
                    "Pushes toward Fraud",
                    "Pushes toward Legitimate"
                )


                st.dataframe(
                    top_shap[
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


                # LIME

                st.subheader(
                    "LIME Local Explanation"
                )


                lime_display = (
                    lime_df.head(10).copy()
                )


                lime_display[
                    "Direction"
                ] = np.where(
                    lime_display[
                        "LIME_Weight"
                    ] >= 0,
                    "Supports Fraud",
                    "Supports Legitimate"
                )


                st.dataframe(
                    lime_display[
                        [
                            "Feature",
                            "LIME_Weight",
                            "Direction"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )


        # ====================================================
        # BATCH
        # ====================================================

        with tab_batch:

            st.subheader(
                "Batch CSV Upload"
            )


            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                key="batch_predict_upload"
            )


            if uploaded_file is not None:

                batch_df = pd.read_csv(
                    uploaded_file
                )


                missing_cols = (
                    set(FEATURE_COLUMNS)
                    -
                    set(batch_df.columns)
                )


                if missing_cols:

                    st.error(
                        "Missing required columns: "
                        f"{sorted(missing_cols)}"
                    )

                else:

                    X_batch = batch_df[
                        FEATURE_COLUMNS
                    ]


                    predictions = (
                        pred_model.predict(
                            X_batch
                        )
                    )


                    probabilities = (
                        pred_model.predict_proba(
                            X_batch
                        )[:, 1]
                    )


                    results_df = (
                        batch_df.copy()
                    )


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


                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )


                    csv_download = (
                        results_df
                        .to_csv(index=False)
                        .encode("utf-8")
                    )


                    st.download_button(
                        "Download Results as CSV",
                        data=csv_download,
                        file_name="fraud_predictions_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


# ============================================================
# 14. REAL-TIME FRAUD DETECTION
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
        "Live Kafka transaction monitoring with SHAP, LIME and adaptive monitoring."
    )

    st.divider()


    transactions = (
        st.session_state.transactions
    )


    if not transactions:

        st.info(
            "Waiting for transactions from Kafka..."
        )

    else:

        latest = transactions[-1]


        prediction = latest.get(
            "prediction",
            "UNKNOWN"
        )


        probability = latest.get(
            "fraud_probability",
            0
        )


        try:

            probability = float(
                probability
            )

            if probability > 1:

                probability /= 100

        except (
            TypeError,
            ValueError
        ):

            probability = 0


        if prediction == "FRAUD":

            st.error(
                f"FRAUD DETECTED - "
                f"Probability: {probability:.2%}"
            )

        elif prediction == "LEGITIMATE":

            st.success(
                f"LEGITIMATE TRANSACTION - "
                f"Fraud Probability: {probability:.2%}"
            )

        else:

            st.warning(
                "Transaction classification unavailable."
            )


        st.divider()


        # ====================================================
        # ADAPTIVE MONITORING
        # ====================================================

        st.subheader(
            "Adaptive Model Monitoring"
        )


        adaptation = assess_adaptation_status(
            transactions
        )


        a1, a2, a3, a4 = st.columns(4)


        with a1:

            st.metric(
                "Recent Transactions",
                adaptation[
                    "recent_transactions"
                ]
            )


        with a2:

            st.metric(
                "Recent Fraud Rate",
                f"{adaptation['recent_fraud_rate']:.2%}"
            )


        with a3:

            st.metric(
                "Historical Baseline",
                f"{adaptation['historical_fraud_rate']:.2%}"
            )


        with a4:

            ratio = adaptation[
                "change_ratio"
            ]


            ratio_display = (
                "High"
                if np.isinf(ratio)
                else f"{ratio:.2f}x"
            )


            st.metric(
                "Behaviour Change",
                ratio_display
            )


        if (
            adaptation["status"]
            ==
            "ADAPTATION REQUIRED"
        ):

            st.warning(
                "⚠️ ADAPTATION REQUIRED"
            )


        elif (
            adaptation["status"]
            ==
            "MONITORING"
        ):

            st.info(
                "🔄 MONITORING"
            )


        else:

            st.success(
                "✓ MODEL BEHAVIOUR STABLE"
            )


        st.caption(
            adaptation["reason"]
        )


        st.divider()


        # ====================================================
        # LATEST TRANSACTION
        # ====================================================

        st.subheader(
            "Latest Transaction"
        )


        st.json(
            latest
        )


        # ====================================================
        # SHAP
        # ====================================================

        if latest.get(
            "top_shap_features"
        ):

            st.subheader(
                "Top SHAP Risk Factors"
            )


            st.dataframe(
                pd.DataFrame(
                    latest[
                        "top_shap_features"
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # LIME
        # ====================================================

        if latest.get(
            "top_lime_features"
        ):

            st.subheader(
                "Real-Time LIME Explanation"
            )


            st.info(
                "LIME provides a local explanation of the latest transaction."
            )


            st.dataframe(
                pd.DataFrame(
                    latest[
                        "top_lime_features"
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )


        st.divider()


        # ====================================================
        # OPERATIONAL FEEDBACK
        # ====================================================

        st.subheader(
            "Operational Review"
        )


        st.write(
            "Record the outcome of the latest transaction review."
        )


        feedback_options = [
            "Confirmed Fraud",
            "False Positive",
            "Legitimate",
            "Requires Investigation"
        ]


        selected_feedback = st.selectbox(
            "Review Outcome",
            feedback_options,
            key="realtime_feedback"
        )


        reviewer_comment = st.text_area(
            "Reviewer Comment",
            key="realtime_feedback_comment"
        )


        if st.button(
            "Submit Operational Feedback",
            use_container_width=True
        ):

            save_feedback(
                latest,
                selected_feedback,
                reviewer_comment
            )


            st.success(
                "Operational feedback recorded successfully."
            )


# ============================================================
# 15. TRANSACTION EXPLORER
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
        "Investigate individual transactions and model decisions."
    )

    st.divider()


    transactions = (
        st.session_state.transactions
    )


    query = st.text_input(
        "Transaction ID",
        placeholder="Enter transaction ID..."
    )


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

            st.dataframe(
                pd.DataFrame(matches),
                use_container_width=True
            )

        else:

            st.warning(
                "No matching transaction found."
            )


    elif transactions:

        st.dataframe(
            pd.DataFrame(
                transactions
            ),
            use_container_width=True
        )

    else:

        st.info(
            "No transaction data available."
        )


# ============================================================
# 16. TRENDS & MONITORING
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
        "Monitor transaction activity and fraud patterns."
    )

    st.divider()


    transactions = (
        st.session_state.transactions
    )


    if transactions:

        df = pd.DataFrame(
            transactions
        )


        if "timestamp" in df.columns:

            df[
                "timestamp"
            ] = pd.to_datetime(
                df[
                    "timestamp"
                ],
                errors="coerce"
            )


            df = df.dropna(
                subset=["timestamp"]
            )


            if not df.empty:

                chart = (
                    df.set_index(
                        "timestamp"
                    )
                    .resample("1min")
                    .size()
                )


                st.subheader(
                    "Transaction Volume"
                )


                st.line_chart(
                    chart
                )

            else:

                st.info(
                    "Valid timestamp information is not available."
                )

        else:

            st.info(
                "Timestamp information is not available yet."
            )

    else:

        st.info(
            "Waiting for transaction data..."
        )


# ============================================================
# 17. THRESHOLD TUNER
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
        "Explore the relationship between fraud probability thresholds and operational classification."
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
        A lower threshold increases sensitivity but may increase false positives.

        A higher threshold may reduce false alarms but can increase missed fraud.

        The final operational threshold should be selected using validation data
        and the relative costs of false positives and false negatives.
        """
    )


# ============================================================
# 18. OPERATIONAL FEEDBACK
# ============================================================

elif current_page == "Operational Feedback":

    st.image(
        TRANSACTION_ICON,
        width=65
    )

    st.title(
        "Operational Feedback"
    )

    st.write(
        "Capture human review outcomes that can support future model adaptation."
    )

    st.divider()


    transactions = (
        st.session_state.transactions
    )


    if not transactions:

        st.info(
            "No real-time transactions are currently available."
        )

    else:

        transaction_options = [
            str(
                t.get(
                    "transaction_id",
                    f"Transaction {i+1}"
                )
            )
            for i, t
            in enumerate(transactions)
        ]


        selected_id = st.selectbox(
            "Select Transaction",
            transaction_options
        )


        selected_transaction = next(
            (
                t
                for t in transactions
                if str(
                    t.get(
                        "transaction_id",
                        ""
                    )
                )
                == selected_id
            ),
            transactions[0]
        )


        st.subheader(
            "Selected Transaction"
        )


        st.json(
            selected_transaction
        )


        feedback_options = [
            "Confirmed Fraud",
            "False Positive",
            "Legitimate",
            "Requires Investigation"
        ]


        feedback_label = st.selectbox(
            "Operational Outcome",
            feedback_options
        )


        comment = st.text_area(
            "Reviewer Comment"
        )


        if st.button(
            "Save Feedback",
            use_container_width=True
        ):

            save_feedback(
                selected_transaction,
                feedback_label,
                comment
            )


            st.success(
                "Feedback saved successfully."
            )


    st.divider()


    st.subheader(
        "Feedback History"
    )


    if st.session_state.feedback_records:

        feedback_df = pd.DataFrame(
            st.session_state.feedback_records
        )


        st.dataframe(
            feedback_df,
            use_container_width=True,
            hide_index=True
        )


        feedback_csv = (
            feedback_df
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            "Download Feedback Dataset",
            data=feedback_csv,
            file_name="operational_feedback.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.info(
            "No operational feedback has been recorded yet."
        )


# ============================================================
# 19. ADAPTIVE MONITORING
# ============================================================

elif current_page == "Adaptive Monitoring":

    st.image(
        MODEL_ICON,
        width=65
    )

    st.title(
        "Adaptive Monitoring & Model Adaptation"
    )

    st.write(
        """
        Monitor changes in incoming transaction behaviour and determine whether
        the fraud detection model requires reassessment or adaptation.
        """
    )

    st.divider()


    transactions = (
        st.session_state.transactions
    )


    adaptation = assess_adaptation_status(
        transactions
    )


    st.subheader(
        "Current Adaptation Status"
    )


    if (
        adaptation["status"]
        ==
        "ADAPTATION REQUIRED"
    ):

        st.warning(
            "⚠️ ADAPTATION REQUIRED"
        )

    elif (
        adaptation["status"]
        ==
        "MONITORING"
    ):

        st.info(
            "🔄 MONITORING"
        )

    else:

        st.success(
            "✓ MODEL BEHAVIOUR STABLE"
        )


    st.write(
        adaptation["reason"]
    )


    a1, a2, a3, a4 = st.columns(4)


    with a1:

        st.metric(
            "Recent Transactions",
            adaptation[
                "recent_transactions"
            ]
        )


    with a2:

        st.metric(
            "Recent Fraud Rate",
            f"{adaptation['recent_fraud_rate']:.2%}"
        )


    with a3:

        st.metric(
            "Historical Fraud Rate",
            f"{adaptation['historical_fraud_rate']:.2%}"
        )


    with a4:

        ratio = adaptation[
            "change_ratio"
        ]


        display_ratio = (
            "High"
            if np.isinf(ratio)
            else f"{ratio:.2f}x"
        )


        st.metric(
            "Behaviour Change",
            display_ratio
        )


    st.divider()


    # ========================================================
    # FEEDBACK STATUS
    # ========================================================

    st.subheader(
        "Operational Feedback Available"
    )


    feedback_count = len(
        st.session_state.feedback_records
    )


    f1, f2 = st.columns(2)


    with f1:

        st.metric(
            "Feedback Records",
            feedback_count
        )


    with f2:

        st.metric(
            "Required for Adaptation",
            MIN_FEEDBACK_FOR_ADAPTATION
        )


    st.divider()


    # ========================================================
    # CREATE ADAPTATION REQUEST
    # ========================================================

    if (
        adaptation["status"]
        ==
        "ADAPTATION REQUIRED"
    ):

        st.subheader(
            "Model Adaptation Request"
        )


        if st.button(
            "Create Adaptation Request",
            use_container_width=True
        ):

            request = (
                create_adaptation_request(
                    adaptation
                )
            )


            st.session_state.adaptation_history.append(
                request
            )


            st.success(
                "Adaptation request created successfully."
            )


            st.json(
                request
            )


        st.info(
            """
            The adaptation request does not automatically replace the deployed model.

            The recommended workflow is:

            1. Review the drift signal.
            2. Review operational feedback.
            3. Create the adaptation dataset.
            4. Retrain/recalibrate a candidate model.
            5. Validate the candidate model.
            6. Approve deployment.
            7. Replace the production model only after validation.
            """
        )


    else:

        st.info(
            "No model adaptation is currently required."
        )


    # ========================================================
    # ADAPTATION DATASET
    # ========================================================

    st.divider()


    st.subheader(
        "Feedback → Model Learning Dataset"
    )


    if st.button(
        "Prepare Adaptation Dataset",
        use_container_width=True
    ):

        adaptation_dataset = (
            create_adaptation_dataset()
        )


        if adaptation_dataset is None:

            st.warning(
                "There is not enough labelled feedback to create an adaptation dataset."
            )

        else:

            st.success(
                "Adaptation dataset created successfully."
            )


            st.dataframe(
                adaptation_dataset,
                use_container_width=True,
                hide_index=True
            )


            st.caption(
                "Saved as adaptation_dataset.csv"
            )


# ============================================================
# 20. COST & VALUE ANALYSIS
# ============================================================

elif current_page == "Cost & Value Analysis":

    st.image(
        VALUE_PROTECTED_ICON,
        width=65
    )

    st.title(
        "Cost / Value Framing"
    )

    st.write(
        """
        Translate fraud detection outcomes into operational and financial indicators.
        Values are configurable assumptions and should be replaced with institution-specific
        estimates when available.
        """
    )

    st.divider()


    # ========================================================
    # ASSUMPTIONS
    # ========================================================

    st.subheader(
        "Cost & Value Assumptions"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        average_fraud_loss = st.number_input(
            "Average Loss per Fraudulent Transaction",
            min_value=0.0,
            value=DEFAULT_AVERAGE_FRAUD_LOSS,
            step=1000.0
        )


    with c2:

        false_alarm_cost = st.number_input(
            "Cost per False Alarm",
            min_value=0.0,
            value=DEFAULT_FALSE_ALARM_COST,
            step=100.0
        )


    with c3:

        investigation_cost = st.number_input(
            "Investigation Cost per Fraud Alert",
            min_value=0.0,
            value=DEFAULT_INVESTIGATION_COST,
            step=100.0
        )


    st.divider()


    # ========================================================
    # CALCULATE
    # ========================================================

    transactions = (
        st.session_state.transactions
    )


    metrics = calculate_cost_value(
        transactions,
        average_fraud_loss,
        false_alarm_cost,
        investigation_cost
    )


    st.subheader(
        "Operational Value Indicators"
    )


    v1, v2, v3, v4 = st.columns(4)


    with v1:

        st.metric(
            "Fraud Caught",
            metrics[
                "fraud_caught"
            ]
        )


    with v2:

        st.metric(
            "Fraud Missed",
            metrics[
                "fraud_missed"
            ]
        )


    with v3:

        st.metric(
            "False Alarms",
            metrics[
                "false_alarms"
            ]
        )


    with v4:

        st.metric(
            "Estimated Value Protected",
            f"₦{metrics['estimated_value_protected']:,.2f}"
        )


    st.divider()


    st.subheader(
        "Estimated Cost Structure"
    )


    k1, k2, k3 = st.columns(3)


    with k1:

        st.metric(
            "False Alarm Cost",
            f"₦{metrics['estimated_false_alarm_cost']:,.2f}"
        )


    with k2:

        st.metric(
            "Investigation Cost",
            f"₦{metrics['estimated_investigation_cost']:,.2f}"
        )


    with k3:

        st.metric(
            "Estimated Net Value",
            f"₦{metrics['estimated_net_value']:,.2f}"
        )


    st.divider()


    st.info(
        """
        These are decision-support estimates, not audited financial figures.

        For the research evaluation, institution-specific estimates can later replace
        the default assumptions to assess the operational value of different model
        thresholds and fraud detection strategies.
        """
    )


# ============================================================
# 21. AI DECISION ASSISTANT
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
        "Interface for interpreting transaction risk and operational model information."
    )

    st.divider()


    question = st.chat_input(
        "Ask about a transaction, prediction or model explanation..."
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
                The decision-support interface is connected to the dashboard data.
                A future language-model integration can use transaction predictions,
                SHAP/LIME explanations, operational feedback and cost/value indicators
                to generate natural-language decision support.
                """
            )


# ============================================================
# 22. FRAMEWORK VALIDATION
# ============================================================

elif current_page == "Framework Validation":

    st.image(
        INFO_ICON,
        width=65
    )

    st.title(
        "Final DSS Integration & Validation"
    )

    st.write(
        """
        Validation of the complete interpretable fraud detection and operational
        decision-support workflow.
        """
    )

    st.divider()


    # ========================================================
    # COMPONENT STATUS
    # ========================================================

    validation_results = []


    validation_results.append(
        {
            "Component":
                "Random Forest Model",
            "Status":
                "AVAILABLE"
                if pred_model is not None
                else "NOT AVAILABLE"
        }
    )


    validation_results.append(
        {
            "Component":
                "SHAP Explanation",
            "Status":
                "AVAILABLE"
                if pred_explainer is not None
                else "NOT AVAILABLE"
        }
    )


    validation_results.append(
        {
            "Component":
                "LIME Explanation",
            "Status":
                "AVAILABLE"
                if lime_explainer is not None
                else "NOT AVAILABLE"
        }
    )


    validation_results.append(
        {
            "Component":
                "Kafka Connection",
            "Status":
                "ONLINE"
                if consumer is not None
                else "OFFLINE"
        }
    )


    validation_results.append(
        {
            "Component":
                "Historical Analysis",
            "Status":
                "AVAILABLE"
                if historical_summary is not None
                else "NOT AVAILABLE"
        }
    )


    validation_results.append(
        {
            "Component":
                "Operational Feedback",
            "Status":
                "READY"
        }
    )


    validation_results.append(
        {
            "Component":
                "Adaptive Monitoring",
            "Status":
                "READY"
        }
    )


    validation_results.append(
        {
            "Component":
                "Cost / Value Framing",
            "Status":
                "READY"
        }
    )


    validation_df = pd.DataFrame(
        validation_results
    )


    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # ========================================================
    # CURRENT ADAPTATION STATUS
    # ========================================================

    adaptation = assess_adaptation_status(
        st.session_state.transactions
    )


    st.subheader(
        "Current Adaptation Status"
    )


    if (
        adaptation["status"]
        ==
        "ADAPTATION REQUIRED"
    ):

        st.warning(
            "ADAPTATION REQUIRED"
        )

    elif (
        adaptation["status"]
        ==
        "MONITORING"
    ):

        st.info(
            "MONITORING"
        )

    else:

        st.success(
            "MODEL BEHAVIOUR STABLE"
        )


    st.write(
        adaptation["reason"]
    )


    st.divider()


    # ========================================================
    # END-TO-END WORKFLOW
    # ========================================================

    st.subheader(
        "End-to-End DSS Workflow"
    )


    workflow_df = pd.DataFrame(
        {
            "Stage": [
                "1. Transaction Ingestion",
                "2. Fraud Prediction",
                "3. SHAP Explanation",
                "4. LIME Explanation",
                "5. Operational Decision",
                "6. Human Feedback",
                "7. Drift Monitoring",
                "8. Adaptation Trigger",
                "9. Adaptation Dataset",
                "10. Model Reassessment",
                "11. Cost / Value Evaluation",
            ],

            "Component": [
                "Apache Kafka",
                "Random Forest",
                "SHAP",
                "LIME",
                "DSS Dashboard",
                "Operational User",
                "Adaptive Monitoring",
                "Adaptation Controller",
                "Feedback Dataset",
                "Model Development Pipeline",
                "Cost / Value Framework",
            ],

            "Status": [
                "CONNECTED",
                "CONNECTED",
                "CONNECTED",
                "CONNECTED",
                "CONNECTED",
                "READY",
                "READY",
                "READY",
                "CONTROLLED PROCESS",
                "READY",
            ]
        }
    )


    st.dataframe(
        workflow_df,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    st.success(
        """
        The dashboard integrates prediction, explainability, real-time monitoring,
        operational feedback, adaptive monitoring and cost/value decision support.

        Model replacement remains a controlled validation activity rather than an
        automatic dashboard action.
        """
    )


# ============================================================
# 23. ABOUT THE FRAMEWORK
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
        ### Development of Adaptive and Interpretable Machine Learning Framework
        for Real-Time Fraud Detection and Operational Decision Support in Digital Banking

        **Developed by: Adeseye Samuel Ademola**
        """
    )


    st.divider()


    st.write(
        """
        The framework integrates machine learning, real-time transaction streaming,
        Explainable Artificial Intelligence, adaptive monitoring, operational feedback
        and decision-support capabilities.
        """
    )


    st.subheader(
        "Core Technologies"
    )


    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Pandas / NumPy",
                "Scikit-learn",
                "Random Forest",
                "Apache Kafka",
                "SHAP",
                "LIME",
                "Streamlit",
            ],

            "Purpose": [
                "Application and ML development",
                "Data processing",
                "Machine learning",
                "Fraud classification",
                "Real-time transaction streaming",
                "Global and local model explainability",
                "Local model explanation",
                "Decision-support interface",
            ],
        }
    )


    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    st.subheader(
        "Model Input"
    )


    st.write(
        """
        The trained fraud detection model uses 30 features:

        **Time, V1-V28 and Amount.**
        """
    )


    st.divider()


    st.subheader(
        "Interpretability"
    )


    st.write(
        """
        **Global interpretability:** Understanding model behaviour across historical
        transaction data using global SHAP importance.

        **Local interpretability:** Understanding individual model decisions using
        SHAP and LIME explanations.
        """
    )


    st.divider()


    st.subheader(
        "Adaptive Learning"
    )


    st.write(
        """
        The framework monitors incoming transaction behaviour against a historical
        baseline. When meaningful behavioural change is detected, the system can
        trigger an adaptation request.

        Operational feedback can then be incorporated into an adaptation dataset
        for controlled model reassessment and retraining.
        """
    )


    st.divider()


    st.subheader(
        "Operational Decision Support"
    )


    st.write(
        """
        The DSS does not replace human decision-making. Instead, it provides
        predictive, explanatory, adaptive and economic information that can support
        transaction investigation, prioritisation and fraud-response decisions.
        """
    )


    st.divider()


    st.caption(
        "Interpretable Fraud Detection & Operational Decision Support System"
    )

    st.caption(
        "Developed by Adeseye Samuel Ademola"
    )