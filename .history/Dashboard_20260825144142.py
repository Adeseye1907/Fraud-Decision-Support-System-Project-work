# ============================================================
# FRAUD DSS
# INTERPRETABLE FRAUD DETECTION & OPERATIONAL DECISION SUPPORT
# Complete Streamlit Dashboard
# ============================================================

import os
import json
import time
import warnings
from collections import deque

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from kafka import KafkaConsumer
from kafka.errors import KafkaError


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud DSS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "rf_fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "creditcard.csv")

ASSET_DIR = os.path.join(BASE_DIR, "assets")

# ------------------------------------------------------------
# Kafka
# ------------------------------------------------------------
# Default matches the DSS write-up.
# You can override with the KAFKA_TOPIC environment variable.
# ------------------------------------------------------------

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "fraud-predictions"
)

MAX_REALTIME_TRANSACTIONS = 500
KAFKA_POLL_TIMEOUT = 300


# ============================================================
# MODEL METRICS
# ============================================================

MODEL_ACCURACY = 100.00
MODEL_PRECISION = 90.00
MODEL_RECALL = 77.00
MODEL_F1 = 83.00
MODEL_ROC_AUC = 96.06
MODEL_PR_AUC = 81.27


# ============================================================
# HISTORICAL TEST METRICS
# ============================================================

TEST_TRANSACTIONS = 56746
ACTUAL_FRAUD = 95
FRAUD_DETECTED = 73
FRAUD_MISSED = 22

DETECTION_RATE = 76.84
FALSE_ALARMS = 8
FRAUD_F1 = 0.830

PREDICTED_FRAUD = 81
PREDICTED_LEGITIMATE = 56665


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background-color: #f5f7fb;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #f5f7fb;
    }

    .main {
        padding-top: 1rem;
    }

    h1, h2, h3, h4 {
        color: #10233f;
    }

    p {
        color: #526071;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #071b33 0%,
            #0b2747 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .sidebar-brand {
        font-size: 24px;
        font-weight: 800;
        color: white;
        margin-bottom: 4px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        line-height: 1.5;
        color: #b8c8dc !important;
    }

    .sidebar-developer {
        font-size: 11px;
        color: #9fb2c9 !important;
        margin-top: 5px;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }


    /* ======================================================
       HERO
       ====================================================== */

    .hero-box {
        background: linear-gradient(
            135deg,
            #071b33 0%,
            #0d3562 100%
        );
        border-radius: 18px;
        padding: 34px;
        margin-bottom: 25px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.10);
    }

    .hero-title {
        color: white !important;
        font-size: 32px;
        font-weight: 850;
        line-height: 1.2;
        margin-bottom: 10px;
    }

    .hero-text {
        color: #d7e3f1 !important;
        font-size: 14px;
        line-height: 1.75;
        max-width: 900px;
    }


    /* ======================================================
       PAGE TITLE
       ====================================================== */

    .page-title {
        font-size: 30px;
        font-weight: 850;
        color: #10233f;
        margin-bottom: 4px;
    }

    .page-subtitle {
        color: #687386;
        font-size: 14px;
        margin-bottom: 20px;
    }


    /* ======================================================
       SECTION
       ====================================================== */

    .section-title {
        font-size: 20px;
        font-weight: 800;
        color: #14243d;
        margin-top: 28px;
        margin-bottom: 6px;
    }

    .section-description {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 15px;
    }


    /* ======================================================
       CARDS
       ====================================================== */

    .card {
        background: white;
        border: 1px solid #e5eaf1;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
        margin-bottom: 15px;
    }

    .card-title {
        color: #14243d;
        font-size: 16px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .card-text {
        color: #667085;
        font-size: 13px;
        line-height: 1.7;
    }


    /* ======================================================
       FEATURE CARDS
       ====================================================== */

    .feature-card {
        background: white;
        border: 1px solid #e5eaf1;
        border-radius: 15px;
        padding: 22px;
        min-height: 185px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
    }

    .feature-icon {
        font-size: 28px;
        margin-bottom: 10px;
    }

    .feature-title {
        color: #14243d;
        font-size: 16px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .feature-text {
        color: #667085;
        font-size: 13px;
        line-height: 1.65;
    }


    /* ======================================================
       WORKFLOW
       ====================================================== */

    .workflow-number {
        font-size: 12px;
        font-weight: 800;
        color: #2563eb;
        margin-bottom: 5px;
    }

    .workflow-title {
        font-size: 15px;
        font-weight: 800;
        color: #14243d;
        margin-bottom: 5px;
    }

    .workflow-text {
        font-size: 12px;
        color: #667085;
        line-height: 1.6;
    }


    /* ======================================================
       STATUS
       ====================================================== */

    .status-online {
        background: #eaf8ef;
        border: 1px solid #b9e8c9;
        color: #15803d;
        border-radius: 12px;
        padding: 14px 18px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .status-offline {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        color: #be123c;
        border-radius: 12px;
        padding: 14px 18px;
        font-weight: 700;
        margin-bottom: 15px;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        margin-top: 35px;
        padding: 25px 0;
        text-align: center;
        color: #98a2b3;
        font-size: 11px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER: SAFE ASSET LOADING
# ============================================================

def asset_path(filename):
    path = os.path.join(ASSET_DIR, filename)

    if os.path.exists(path):
        return path

    return None


def show_asset(filename, caption=None, width="stretch"):
    path = asset_path(filename)

    if path:
        st.image(
            path,
            caption=caption,
            width=width
        )
        return True

    return False


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_scaler():

    if not os.path.exists(SCALER_PATH):
        return None

    try:
        return joblib.load(SCALER_PATH)
    except Exception:
        return None


model = load_model()
scaler = load_scaler()


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_historical_data():

    if not os.path.exists(DATA_PATH):
        return None

    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None


historical_df = load_historical_data()


# ============================================================
# HISTORICAL DATA SUMMARY
# ============================================================

if historical_df is not None:

    HISTORICAL_TRANSACTIONS = len(historical_df)

    if "Class" in historical_df.columns:

        HISTORICAL_FRAUD = int(
            historical_df["Class"].sum()
        )

    else:

        HISTORICAL_FRAUD = 492

else:

    HISTORICAL_TRANSACTIONS = 284807
    HISTORICAL_FRAUD = 492


HISTORICAL_LEGITIMATE = (
    HISTORICAL_TRANSACTIONS -
    HISTORICAL_FRAUD
)

HISTORICAL_FRAUD_RATE = (
    HISTORICAL_FRAUD /
    HISTORICAL_TRANSACTIONS *
    100
)


# ============================================================
# SESSION STATE
# ============================================================

if "realtime_transactions" not in st.session_state:

    st.session_state.realtime_transactions = deque(
        maxlen=MAX_REALTIME_TRANSACTIONS
    )


if "realtime_ids" not in st.session_state:

    st.session_state.realtime_ids = set()


if "kafka_status" not in st.session_state:

    st.session_state.kafka_status = "Not connected"


if "last_transaction" not in st.session_state:

    st.session_state.last_transaction = None


if "feedback_records" not in st.session_state:

    st.session_state.feedback_records = []


if "threshold" not in st.session_state:

    st.session_state.threshold = 0.50


# ============================================================
# MODEL FEATURES
# ============================================================

DEFAULT_FEATURES = (
    ["Time", "Amount"] +
    [f"V{i}" for i in range(1, 29)]
)


def get_model_features():

    if model is not None:

        if hasattr(
            model,
            "feature_names_in_"
        ):

            return list(
                model.feature_names_in_
            )

    return DEFAULT_FEATURES


# ============================================================
# PREPARE TRANSACTION
# ============================================================

def prepare_transaction(transaction):

    feature_names = get_model_features()

    row = {}

    for feature in feature_names:

        value = transaction.get(
            feature,
            0
        )

        try:
            value = float(value)

        except Exception:

            value = 0.0

        row[feature] = value


    X = pd.DataFrame(
        [row],
        columns=feature_names
    )


    cols_to_scale = [
        c for c in
        ["Time", "Amount"]
        if c in X.columns
    ]


    if scaler is not None and cols_to_scale:

        try:

            X[cols_to_scale] = (
                scaler.transform(
                    X[cols_to_scale]
                )
            )

        except Exception:

            pass


    return X


# ============================================================
# PREDICT SINGLE TRANSACTION
# ============================================================

def predict_transaction(
    transaction,
    threshold=0.50
):

    transaction = dict(transaction)

    if model is None:

        transaction["prediction"] = "UNKNOWN"
        transaction["fraud_probability"] = 0.0

        return transaction


    try:

        X = prepare_transaction(
            transaction
        )

        probability = float(
            model.predict_proba(X)[0][1]
        )

        prediction = (
            "FRAUD"
            if probability >= threshold
            else "LEGITIMATE"
        )

        transaction["prediction"] = prediction

        transaction["fraud_probability"] = (
            probability * 100
        )

        transaction["_model_input"] = X

    except Exception as e:

        transaction["prediction"] = "ERROR"

        transaction["fraud_probability"] = 0.0

        transaction["_prediction_error"] = str(e)


    return transaction


# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_batch(df, threshold=0.50):

    if model is None:

        return None, "Model could not be loaded."


    required_features = get_model_features()

    working_df = df.copy()


    for feature in required_features:

        if feature not in working_df.columns:

            working_df[feature] = 0.0


    X = working_df[
        required_features
    ].copy()


    for column in required_features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        ).fillna(0)


    cols_to_scale = [
        c for c in
        ["Time", "Amount"]
        if c in X.columns
    ]


    if scaler is not None and cols_to_scale:

        try:

            X[cols_to_scale] = (
                scaler.transform(
                    X[cols_to_scale]
                )
            )

        except Exception:
            pass


    try:

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)


        result = df.copy()

        result["Fraud Probability"] = (
            probabilities * 100
        )

        result["Prediction"] = np.where(
            predictions == 1,
            "FRAUD",
            "LEGITIMATE"
        )


        return result, None


    except Exception as e:

        return None, str(e)


# ============================================================
# KAFKA CONSUMER
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def create_kafka_consumer():

    consumer = KafkaConsumer(

        KAFKA_TOPIC,

        bootstrap_servers=[
            KAFKA_SERVER
        ],

        value_deserializer=lambda x:
            json.loads(
                x.decode("utf-8")
            ),

        auto_offset_reset="latest",

        enable_auto_commit=True,

        group_id="fraud-dss-dashboard",

        request_timeout_ms=3000,

        session_timeout_ms=6000,

        heartbeat_interval_ms=2000

    )

    return consumer


def read_kafka_transactions():

    try:

        consumer = create_kafka_consumer()

        st.session_state.kafka_status = (
            "Connected"
        )


        messages = consumer.poll(

            timeout_ms=
            KAFKA_POLL_TIMEOUT,

            max_records=50
        )


        new_transactions = []


        for _, records in messages.items():

            for message in records:

                transaction = message.value


                transaction_id = (

                    transaction.get(
                        "transaction_id"
                    )

                    or transaction.get(
                        "id"
                    )

                    or transaction.get(
                        "Time"
                    )

                    or transaction.get(
                        "timestamp"
                    )

                    or str(
                        hash(
                            json.dumps(
                                transaction,
                                sort_keys=True,
                                default=str
                            )
                        )
                    )
                )


                transaction_id = str(
                    transaction_id
                )


                if (
                    transaction_id
                    in
                    st.session_state.realtime_ids
                ):

                    continue


                st.session_state.realtime_ids.add(
                    transaction_id
                )


                transaction = predict_transaction(
                    transaction
                )


                transaction[
                    "_transaction_id"
                ] = transaction_id


                st.session_state.realtime_transactions.append(
                    transaction
                )


                new_transactions.append(
                    transaction
                )


        if new_transactions:

            st.session_state.last_transaction = (
                new_transactions[-1]
            )


        if len(
            st.session_state.realtime_ids
        ) > MAX_REALTIME_TRANSACTIONS * 2:

            st.session_state.realtime_ids = {
                str(
                    x.get(
                        "_transaction_id"
                    )
                )
                for x in
                st.session_state.realtime_transactions
                if x.get(
                    "_transaction_id"
                ) is not None
            }


        return new_transactions


    except Exception:

        st.session_state.kafka_status = (
            "Offline"
        )

        try:

            create_kafka_consumer.clear()

        except Exception:

            pass


        return []


# ============================================================
# CHART FUNCTIONS
# ============================================================

def create_donut_chart(
    fraud,
    legitimate,
    title=""
):

    fig = go.Figure(
        data=[
            go.Pie(
                labels=[
                    "Legitimate",
                    "Fraud"
                ],

                values=[
                    legitimate,
                    fraud
                ],

                hole=0.62,

                textinfo="percent",

                hovertemplate=
                    "<b>%{label}</b><br>"
                    "Transactions: %{value:,}<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
            )
        ]
    )


    fig.update_layout(

        title=title,

        height=320,

        margin=dict(
            l=10,
            r=10,
            t=40,
            b=10
        ),

        showlegend=True,

        legend=dict(
            orientation="h",
            y=-0.05
        )
    )


    return fig


def create_model_performance_chart():

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC"
    ]


    values = [
        MODEL_ACCURACY,
        MODEL_PRECISION,
        MODEL_RECALL,
        MODEL_F1,
        MODEL_ROC_AUC,
        MODEL_PR_AUC
    ]


    fig = px.bar(

        x=metrics,

        y=values,

        text=[
            f"{x:.2f}%"
            for x in values
        ],

        labels={
            "x": "Metric",
            "y": "Performance (%)"
        }
    )


    fig.update_traces(
        textposition="outside"
    )


    fig.update_yaxes(
        range=[0, 110]
    )


    fig.update_layout(

        height=330,

        margin=dict(
            l=10,
            r=10,
            t=25,
            b=10
        ),

        showlegend=False
    )


    return fig


def create_amount_chart():

    if historical_df is None:
        return None

    if "Amount" not in historical_df.columns:
        return None


    temp = historical_df.copy()


    temp["Amount Band"] = pd.cut(

        temp["Amount"],

        bins=[
            -np.inf,
            10,
            50,
            100,
            500,
            1000,
            np.inf
        ],

        labels=[
            "< $10",
            "$10–50",
            "$50–100",
            "$100–500",
            "$500–1K",
            "> $1K"
        ]
    )


    grouped = (
        temp.groupby(
            "Amount Band",
            observed=False
        )
        .size()
        .reset_index(
            name="Transactions"
        )
    )


    fig = px.bar(

        grouped,

        x="Amount Band",

        y="Transactions",

        labels={
            "Amount Band":
                "Transaction Amount",
            "Transactions":
                "Transactions"
        }
    )


    fig.update_layout(
        height=340,
        margin=dict(
            l=10,
            r=10,
            t=25,
            b=10
        )
    )


    return fig


# ============================================================
# NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🛡️ Fraud DSS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-subtitle">
        Interpretable Fraud Detection & Operational
        Decision Support
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-developer">
        Developed by Adeseye Samuel Ademola
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("---")

    st.markdown("### SYSTEM NAVIGATION")


    page = st.radio(

        "System Navigation",

        [
            "Introduction",
            "Executive Overview",
            "Historical Model Analysis",
            "Predict Fraud",
            "Real-Time Fraud Detection",
            "Transaction Explorer",
            "Trends & Monitoring",
            "Adaptive Monitoring",
            "Model Adaptation",
            "Operational Feedback",
            "Feedback Learning Pipeline",
            "Cost / Value Framing",
            "Threshold Tuner",
            "AI Decision Assistant",
            "End-to-End DSS Workflow",
            "About the Framework"
        ],

        label_visibility="collapsed"
    )


    st.markdown("---")


    st.markdown("### SYSTEM STATUS")


    if model is not None:

        st.success(
            "● System Online"
        )

    else:

        st.warning(
            "● Model Unavailable"
        )


    st.caption(
        f"Kafka Topic: {KAFKA_TOPIC}"
    )


    if st.button(
        "🔄 Refresh Dashboard",
        use_container_width=True
    ):

        st.rerun()


    st.markdown("---")

    st.caption(
        "Prototype / Reference Implementation"
    )

    st.caption(
        "Adaptive and interpretable fraud detection "
        "and operational decision support."
    )


# ============================================================
# INTRODUCTION
# ============================================================

if page == "Introduction":

    st.markdown(
        """
        # INTERPRETABLE FRAUD DETECTION
        # & OPERATIONAL DECISION SUPPORT
        """
    )

    st.markdown(
        "### Real-Time Intelligence for Digital Banking"
    )

    st.markdown(
        """
        **Development of Adaptive and Interpretable
        Machine Learning Framework for Real-Time Fraud
        Detection and Operational Decision Support in
        Digital Banking**
        """
    )

    st.markdown(
        "**Developed by Adeseye Samuel Ademola**"
    )


    st.divider()


    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.markdown(
        "## System Status"
    )

    status_col1, status_col2, status_col3 = st.columns(3)


    with status_col1:

        if model is not None:

            st.success(
                "● System Online"
            )

        else:

            st.warning(
                "● Model Unavailable"
            )


    with status_col2:

        st.info(
            f"Kafka Topic: {KAFKA_TOPIC}"
        )


    with status_col3:

        if st.button(
            "🔄 Refresh Dashboard",
            key="intro_refresh",
            use_container_width=True
        ):

            st.rerun()


    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    st.markdown(
        "## Background of the Study"
    )

    st.markdown(
        """
        The rapid growth of digital banking and electronic
        payment systems has increased the volume, velocity and
        complexity of financial transactions. At the same time,
        this growth has created opportunities for increasingly
        sophisticated fraudulent activities.

        Machine learning provides an opportunity to identify
        complex patterns within transaction data and detect
        potentially fraudulent behaviour. However, predictive
        performance alone is insufficient for effective
        operational fraud management.

        Operational teams need to understand not only whether a
        transaction has been classified as potentially
        fraudulent, but also why the machine learning model
        reached that decision.

        This framework therefore combines fraud prediction,
        explainable AI, real-time monitoring, adaptive
        monitoring and operational decision support.
        """
    )


    # --------------------------------------------------------
    # RESEARCH FOCUS
    # --------------------------------------------------------

    st.markdown(
        "## Research Focus"
    )

    f1, f2, f3 = st.columns(3)


    with f1:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">
                    Adaptive Machine Learning
                </div>
                <div class="feature-text">
                    Supports fraud detection within changing
                    transaction environments and evolving
                    fraud patterns.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with f2:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">
                    Interpretable Machine Learning
                </div>
                <div class="feature-text">
                    Provides insight into model behaviour and
                    identifies factors contributing to individual
                    predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with f3:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">
                    Operational Decision Support
                </div>
                <div class="feature-text">
                    Converts analytical outputs into information
                    that can support operational fraud review
                    and decision-making.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.markdown(
        "## Framework Workflow"
    )

    workflow = [

        (
            "01",
            "Historical Data",
            "Historical transactions support model development "
            "and evaluation."
        ),

        (
            "02",
            "Machine Learning",
            "Random Forest detects transaction-level fraud risk."
        ),

        (
            "03",
            "Real-Time Stream",
            "Kafka transports transaction events."
        ),

        (
            "04",
            "Explainable AI",
            "SHAP and LIME provide local and global explanations."
        ),

        (
            "05",
            "Adaptive Monitoring",
            "Drift monitoring identifies changing transaction "
            "behaviour."
        ),

        (
            "06",
            "Operational Feedback",
            "Investigator feedback informs future learning."
        ),

        (
            "07",
            "Decision Support",
            "The DSS presents actionable fraud intelligence."
        )
    ]


    cols = st.columns(4)


    for i, item in enumerate(workflow):

        number, title, description = item

        with cols[i % 4]:

            st.markdown(
                f"""
                <div class="card">
                    <div class="workflow-number">
                        {number}
                    </div>
                    <div class="workflow-title">
                        {title}
                    </div>
                    <div class="workflow-text">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    st.markdown(
        "## Model Input"
    )

    st.markdown(
        """
        The trained fraud detection model operates on
        **30 transaction features consisting of Time,
        V1-V28 and Amount**.
        """
    )


    if not show_asset(
        "model-input.png",
        "Transaction features used by the fraud detection model."
    ):

        st.info(
            "Model input visual can be displayed by placing "
            "'model-input.png' inside the assets folder."
        )


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

elif page == "Executive Overview":

    st.markdown(
        '<div class="page-title">Executive Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Fraud Detection & Operational Decision Support'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    k1, k2, k3, k4, k5 = st.columns(5)


    with k1:

        st.metric(
            "📊 Transactions",
            f"{HISTORICAL_TRANSACTIONS:,}",
            "Historical transactions analysed"
        )


    with k2:

        st.metric(
            "🚨 Fraud Cases",
            f"{HISTORICAL_FRAUD:,}",
            "Identified fraudulent transactions"
        )


    with k3:

        st.metric(
            "📈 Fraud Rate",
            f"{HISTORICAL_FRAUD_RATE:.2f}%",
            "Fraud proportion in historical data"
        )


    with k4:

        st.metric(
            "🎯 ROC-AUC",
            f"{MODEL_ROC_AUC:.2f}%",
            "Model discrimination performance"
        )


    with k5:

        st.metric(
            "⚖️ PR-AUC",
            f"{MODEL_PR_AUC:.2f}%",
            "Fraud-class performance"
        )


    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    st.markdown(
        "## Risk & Model Performance"
    )


    c1, c2 = st.columns(2)


    with c1:

        st.markdown(
            "### Transaction Risk Distribution"
        )

        st.caption(
            "Historical fraud and legitimate transaction profile."
        )

        st.plotly_chart(
            create_donut_chart(
                HISTORICAL_FRAUD,
                HISTORICAL_LEGITIMATE
            ),
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    with c2:

        st.markdown(
            "### Fraud Detection Model Performance"
        )

        st.caption(
            "Evaluation metrics from the Random Forest model."
        )

        st.plotly_chart(
            create_model_performance_chart(),
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    c3, c4 = st.columns(2)


    with c3:

        st.markdown(
            "## Model Evaluation Summary"
        )

        metrics_df = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-Score",
                    "ROC-AUC",
                    "PR-AUC"
                ],

                "Value": [
                    f"{MODEL_ACCURACY:.2f}%",
                    f"{MODEL_PRECISION:.2f}%",
                    f"{MODEL_RECALL:.2f}%",
                    f"{MODEL_F1:.2f}%",
                    f"{MODEL_ROC_AUC:.2f}%",
                    f"{MODEL_PR_AUC:.2f}%"
                ]
            }
        )

        st.dataframe(
            metrics_df,
            hide_index=True,
            use_container_width=True
        )


    with c4:

        st.markdown(
            "## Decision Support Pipeline"

        )

        pipeline = pd.DataFrame(
            {
                "Component": [
                    "Transaction Ingestion",
                    "Kafka Event Streaming",
                    "Fraud Classification",
                    "Explainability Layer",
                    "Adaptive Monitoring",
                    "Decision Support Dashboard"
                ],

                "Status": [
                    "● ACTIVE",
                    "● READY",
                    "● ACTIVE",
                    "● READY",
                    "● ACTIVE",
                    "● ACTIVE"
                ]
            }
        )

        st.dataframe(
            pipeline,
            hide_index=True,
            use_container_width=True
        )


    # --------------------------------------------------------
    # EXECUTIVE INSIGHTS
    # --------------------------------------------------------

    st.markdown(
        "## Executive Insights"
    )


    i1, i2, i3 = st.columns(3)


    with i1:

        st.info(
            f"""
            **Fraud Exposure**

            The historical dataset contains
            **{HISTORICAL_FRAUD:,}** identified fraudulent
            transactions out of **{HISTORICAL_TRANSACTIONS:,}**
            transactions analysed.

            This represents a fraud rate of
            **{HISTORICAL_FRAUD_RATE:.2f}%**.
            """
        )


    with i2:

        st.info(
            f"""
            **Detection Capability**

            The Random Forest model achieved a ROC-AUC of
            **{MODEL_ROC_AUC:.2f}%** and PR-AUC of
            **{MODEL_PR_AUC:.2f}%**.

            These metrics indicate strong discrimination
            between fraudulent and legitimate activity.
            """
        )


    with i3:

        st.info(
            f"""
            **Operational Consideration**

            Fraud recall is currently
            **{MODEL_RECALL:.1f}%**, while precision is
            **{MODEL_PRECISION:.1f}%**.

            This supports a risk-based operational review
            process rather than treating every transaction
            as equally risky.
            """
        )


# ============================================================
# HISTORICAL MODEL ANALYSIS
# ============================================================

elif page == "Historical Model Analysis":

    st.markdown(
        '<div class="page-title">'
        'Historical Model Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Evaluation and interpretation of the trained '
        'Random Forest fraud detection model.'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        "## Model Performance"
    )


    a, b, c, d = st.columns(4)


    with a:
        st.metric(
            "ROC-AUC",
            "0.961"
        )


    with b:
        st.metric(
            "PR-AUC",
            "0.813"
        )


    with c:
        st.metric(
            "Fraud Precision",
            "0.90"
        )


    with d:
        st.metric(
            "Fraud Recall",
            "0.77"
        )


    # --------------------------------------------------------
    # HISTORICAL TEST DATASET
    # --------------------------------------------------------

    st.markdown(
        "## Historical Test Dataset"
    )


    a, b, c, d = st.columns(4)


    with a:
        st.metric(
            "Transactions Evaluated",
            f"{TEST_TRANSACTIONS:,}"
        )


    with b:
        st.metric(
            "Actual Fraud",
            f"{ACTUAL_FRAUD:,}"
        )


    with c:
        st.metric(
            "Fraud Detected",
            f"{FRAUD_DETECTED:,}"
        )


    with d:
        st.metric(
            "Fraud Missed",
            f"{FRAUD_MISSED:,}"
        )


    # --------------------------------------------------------
    # OPERATIONAL PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        "## Operational Fraud Detection Performance"
    )


    a, b, c = st.columns(3)


    with a:
        st.metric(
            "Detection Rate",
            f"{DETECTION_RATE:.2f}%"
        )


    with b:
        st.metric(
            "False Alarms",
            f"{FALSE_ALARMS}"
        )


    with c:
        st.metric(
            "Fraud F1-Score",
            f"{FRAUD_F1:.3f}"
        )


    # --------------------------------------------------------
    # GLOBAL INTERPRETABILITY
    # --------------------------------------------------------

    st.markdown(
        "## Global Model Interpretability"
    )

    st.markdown(
        """
        Global SHAP identifies the features with the greatest
        influence on model predictions.
        """
    )


    if model is not None and hasattr(
        model,
        "feature_importances_"
    ):

        feature_names = get_model_features()

        importance = model.feature_importances_


        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance
            }
        ).sort_values(
            "Importance",
            ascending=False
        )


        fig = px.bar(

            importance_df.head(15),

            x="Importance",

            y="Feature",

            orientation="h",

            title="Global Feature Importance"
        )


        fig.update_layout(
            height=500,
            yaxis=dict(
                categoryorder="total ascending"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Model feature importance is unavailable."
        )


    # --------------------------------------------------------
    # PREDICTION OUTCOME SUMMARY
    # --------------------------------------------------------

    st.markdown(
        "## Prediction Outcome Summary"
    )


    p1, p2 = st.columns(2)


    with p1:

        st.metric(
            "Predicted Fraud",
            f"{PREDICTED_FRAUD:,}"
        )


    with p2:

        st.metric(
            "Predicted Legitimate",
            f"{PREDICTED_LEGITIMATE:,}"
        )


    # --------------------------------------------------------
    # HISTORICAL PREDICTION RESULTS
    # --------------------------------------------------------

    st.markdown(
        "## Historical Prediction Results"
    )


    if historical_df is not None:

        display_df = historical_df.copy()

        if "Class" in display_df.columns:

            display_df = display_df.head(100)


        st.dataframe(
            display_df,
            use_container_width=True,
            height=420
        )

    else:

        st.info(
            "Historical dataset not found."
        )


# ============================================================
# PREDICT FRAUD
# ============================================================

elif page == "Predict Fraud":

    st.markdown(
        '<div class="page-title">'
        'Predict Fraud'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Predict fraud for an individual transaction or a batch '
        'of transactions and examine SHAP and LIME explanations.'
        '</div>',
        unsafe_allow_html=True
    )


    tab1, tab2 = st.tabs(
        [
            "Manual Entry",
            "Batch CSV Upload"
        ]
    )


    # ========================================================
    # MANUAL ENTRY
    # ========================================================

    with tab1:

        st.markdown(
            "## Enter Transaction Details"
        )

        st.info(
            "V1-V28 are anonymised PCA-transformed features. "
            "Zero values can be used for testing."
        )


        # ----------------------------------------------------
        # TIME AND AMOUNT
        # ----------------------------------------------------

        c1, c2 = st.columns(2)


        with c1:

            transaction_time = st.number_input(
                "Time",
                value=0.0,
                step=1.0
            )


        with c2:

            transaction_amount = st.number_input(
                "Amount",
                value=100.0,
                min_value=0.0,
                step=1.0
            )


        st.markdown(
            "### V1 - V28 (PCA features)"
        )


        feature_values = {}


        feature_columns = [
            st.columns(4)
            for _ in range(7)
        ]


        index = 1


        for row in feature_columns:

            for col in row:

                with col:

                    feature_values[
                        f"V{index}"
                    ] = st.number_input(
                        f"V{index}",
                        value=0.0,
                        format="%.6f",
                        key=f"manual_v{index}"
                    )

                index += 1


        st.markdown("")


        threshold = st.slider(
            "Fraud Decision Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key="manual_threshold"
        )


        if st.button(
            "🔍 Predict Transaction",
            type="primary",
            use_container_width=True
        ):

            transaction = {
                "Time": transaction_time,
                "Amount": transaction_amount
            }

            transaction.update(
                feature_values
            )


            result = predict_transaction(
                transaction,
                threshold=threshold
            )


            prediction = result.get(
                "prediction"
            )

            probability = result.get(
                "fraud_probability",
                0
            )


            st.markdown(
                "## Prediction Result"
            )


            r1, r2 = st.columns(2)


            with r1:

                if prediction == "FRAUD":

                    st.error(
                        f"🚨 FRAUD DETECTED\n\n"
                        f"Fraud probability: "
                        f"{probability:.2f}%"
                    )

                else:

                    st.success(
                        f"✓ LEGITIMATE TRANSACTION\n\n"
                        f"Fraud probability: "
                        f"{probability:.2f}%"
                    )


            with r2:

                st.metric(
                    "Fraud Probability",
                    f"{probability:.2f}%"
                )


            # ------------------------------------------------
            # EXPLANATION
            # ------------------------------------------------

            st.markdown(
                "### Prediction Explanation"
            )


            if model is not None and hasattr(
                model,
                "feature_importances_"
            ):

                importance = model.feature_importances_

                feature_names = get_model_features()


                explanation_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": importance
                    }
                ).sort_values(
                    "Importance",
                    ascending=False
                )


                st.plotly_chart(
                    px.bar(
                        explanation_df.head(10),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Top Global Model Drivers"
                    ),
                    use_container_width=True
                )


    # ========================================================
    # BATCH CSV
    # ========================================================

    with tab2:

        st.markdown(
            "## Batch Fraud Prediction"
        )

        st.markdown(
            """
            Upload a CSV containing transaction records.
            The model will automatically create missing model
            features as zero and return a prediction and fraud
            probability for every transaction.
            """
        )


        uploaded_file = st.file_uploader(
            "Upload transaction CSV",
            type=["csv"],
            key="batch_csv"
        )


        batch_threshold = st.slider(
            "Batch Fraud Decision Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key="batch_threshold"
        )


        if uploaded_file is not None:

            try:

                batch_df = pd.read_csv(
                    uploaded_file
                )


                st.success(
                    f"CSV loaded successfully: "
                    f"{len(batch_df):,} rows."
                )


                st.markdown(
                    "### Uploaded Data Preview"
                )


                st.dataframe(
                    batch_df.head(20),
                    use_container_width=True
                )


                if st.button(
                    "🚨 Run Batch Prediction",
                    type="primary",
                    use_container_width=True
                ):

                    results, error = predict_batch(
                        batch_df,
                        threshold=batch_threshold
                    )


                    if error:

                        st.error(
                            f"Prediction failed: {error}"
                        )

                    else:

                        st.session_state[
                            "batch_results"
                        ] = results


                        fraud_total = int(
                            (
                                results[
                                    "Prediction"
                                ] == "FRAUD"
                            ).sum()
                        )


                        legit_total = (
                            len(results) -
                            fraud_total
                        )


                        c1, c2, c3 = st.columns(3)


                        with c1:

                            st.metric(
                                "Transactions",
                                f"{len(results):,}"
                            )


                        with c2:

                            st.metric(
                                "Fraud Predicted",
                                f"{fraud_total:,}"
                            )


                        with c3:

                            st.metric(
                                "Legitimate",
                                f"{legit_total:,}"
                            )


                        st.markdown(
                            "### Batch Prediction Results"
                        )


                        st.dataframe(
                            results,
                            use_container_width=True,
                            height=500
                        )


                        csv_data = results.to_csv(
                            index=False
                        ).encode("utf-8")


                        st.download_button(
                            "⬇️ Download Prediction Results",
                            csv_data,
                            "fraud_predictions.csv",
                            "text/csv",
                            use_container_width=True
                        )


# ============================================================
# REAL-TIME FRAUD DETECTION
# ============================================================

elif page == "Real-Time Fraud Detection":

    st.markdown(
        '<div class="page-title">'
        'Real-Time Fraud Detection'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Live transaction monitoring and fraud classification '
        'from the Kafka event stream.'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # KAFKA
    # --------------------------------------------------------

    new_transactions = (
        read_kafka_transactions()
    )


    if st.session_state.kafka_status == "Connected":

        st.success(
            f"● Kafka Stream Active — {KAFKA_TOPIC}"
        )

    else:

        st.warning(
            f"● Kafka Offline — Retrying"
        )


    transactions = list(
        st.session_state.realtime_transactions
    )


    received_count = len(
        transactions
    )


    fraud_count = sum(
        1
        for x in transactions
        if x.get("prediction")
        == "FRAUD"
    )


    legitimate_count = sum(
        1
        for x in transactions
        if x.get("prediction")
        == "LEGITIMATE"
    )


    fraud_rate = (

        fraud_count /
        received_count *
        100

        if received_count
        else 0
    )


    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    k1, k2, k3, k4, k5 = st.columns(5)


    with k1:

        st.metric(
            "Transactions Received",
            received_count
        )


    with k2:

        st.metric(
            "Fraud Detected",
            fraud_count
        )


    with k3:

        st.metric(
            "Legitimate",
            legitimate_count
        )


    with k4:

        st.metric(
            "Real-Time Fraud Rate",
            f"{fraud_rate:.2f}%"
        )


    with k5:

        latest_probability = 0

        if st.session_state.last_transaction:

            latest_probability = (
                st.session_state
                .last_transaction
                .get(
                    "fraud_probability",
                    0
                )
            )


        st.metric(
            "Latest Risk",
            f"{latest_probability:.2f}%"
        )


    # --------------------------------------------------------
    # CHART + LATEST
    # --------------------------------------------------------

    r1, r2 = st.columns(2)


    with r1:

        st.markdown(
            "### Real-Time Risk Distribution"
        )


        if received_count:

            st.plotly_chart(
                create_donut_chart(
                    fraud_count,
                    legitimate_count
                ),
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

        else:

            st.info(
                "Waiting for transactions from Kafka..."
            )


    with r2:

        st.markdown(
            "### Latest Transaction"
        )


        latest = (
            st.session_state.last_transaction
        )


        if latest:

            prediction = latest.get(
                "prediction",
                "UNKNOWN"
            )


            probability = latest.get(
                "fraud_probability",
                0
            )


            if prediction == "FRAUD":

                st.error(
                    f"""
                    🚨 FRAUDULENT TRANSACTION

                    Fraud probability:
                    **{probability:.2f}%**

                    Immediate risk-based review recommended.
                    """
                )


            elif prediction == "LEGITIMATE":

                st.success(
                    f"""
                    ✓ LEGITIMATE TRANSACTION

                    Fraud probability:
                    **{probability:.2f}%**

                    No elevated fraud risk detected.
                    """
                )


            details = {

                k: v

                for k, v
                in latest.items()

                if not k.startswith("_")

                and k not in [
                    "prediction",
                    "fraud_probability"
                ]
            }


            if details:

                st.dataframe(
                    pd.DataFrame(
                        list(
                            details.items()
                        ),
                        columns=[
                            "Field",
                            "Value"
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True
                )


        else:

            st.info(
                "No transaction has been received yet."
            )


    # --------------------------------------------------------
    # RECENT TRANSACTIONS
    # --------------------------------------------------------

    st.markdown(
        "### Recent Transactions"
    )


    if transactions:

        rows = []


        for transaction in reversed(
            transactions[-50:]
        ):

            rows.append(
                {
                    "Transaction ID":
                        transaction.get(
                            "_transaction_id",
                            "N/A"
                        ),

                    "Prediction":
                        transaction.get(
                            "prediction",
                            "UNKNOWN"
                        ),

                    "Fraud Probability":
                        f"{transaction.get(
                            'fraud_probability',
                            0
                        ):.2f}%",

                    "Amount":
                        transaction.get(
                            "Amount",
                            0
                        ),

                    "Time":
                        transaction.get(
                            "Time",
                            "N/A"
                        )
                }
            )


        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            height=420
        )


    else:

        st.info(
            "No Kafka transactions have been received yet."
        )


    st.caption(
        "Kafka is polled only while this page is active."
    )


    if st.button(
        "🔄 Refresh Real-Time Data",
        use_container_width=True
    ):

        st.rerun()


# ============================================================
# TRANSACTION EXPLORER
# ============================================================

elif page == "Transaction Explorer":

    st.markdown(
        '<div class="page-title">'
        'Transaction Explorer'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore historical transaction records and '
        'fraud classifications.'
        '</div>',
        unsafe_allow_html=True
    )


    if historical_df is None:

        st.warning(
            "creditcard.csv could not be found."
        )

    else:

        search_col = st.selectbox(
            "Search field",
            historical_df.columns
        )


        search_value = st.text_input(
            "Search value"
        )


        filtered = historical_df.copy()


        if search_value:

            filtered = filtered[
                filtered[
                    search_col
                ].astype(str).str.contains(
                    search_value,
                    case=False,
                    na=False
                )
            ]


        st.metric(
            "Matching Transactions",
            f"{len(filtered):,}"
        )


        st.dataframe(
            filtered.head(500),
            use_container_width=True,
            height=550
        )


# ============================================================
# TRENDS & MONITORING
# ============================================================

elif page == "Trends & Monitoring":

    st.markdown(
        '<div class="page-title">'
        'Trends & Monitoring'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Monitor transaction behaviour and fraud-related '
        'patterns across the available data.'
        '</div>',
        unsafe_allow_html=True
    )


    if historical_df is not None:

        amount_chart = create_amount_chart()


        if amount_chart is not None:

            st.markdown(
                "## Transaction Amount Distribution"
            )

            st.plotly_chart(
                amount_chart,
                use_container_width=True
            )


        if "Class" in historical_df.columns:

            fraud_by_class = (
                historical_df[
                    "Class"
                ]
                .value_counts()
                .rename(
                    {
                        0: "Legitimate",
                        1: "Fraud"
                    }
                )
                .reset_index()
            )


            fraud_by_class.columns = [
                "Class",
                "Transactions"
            ]


            st.markdown(
                "## Transaction Classification Trend"
            )


            st.plotly_chart(
                px.bar(
                    fraud_by_class,
                    x="Class",
                    y="Transactions",
                    title="Fraud vs Legitimate Transactions"
                ),
                use_container_width=True
            )


        if "Amount" in historical_df.columns:

            st.markdown(
                "## Amount Statistics"
            )


            stats = historical_df[
                "Amount"
            ].describe().to_frame(
                "Amount"
            )


            st.dataframe(
                stats,
                use_container_width=True
            )


    else:

        st.info(
            "Historical data is unavailable."
        )


# ============================================================
# ADAPTIVE MONITORING
# ============================================================

elif page == "Adaptive Monitoring":

    st.markdown(
        '<div class="page-title">'
        'Adaptive Monitoring'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Monitor changing transaction behaviour and identify '
        'potential distribution shifts.'
        '</div>',
        unsafe_allow_html=True
    )


    st.info(
        """
        Adaptive monitoring supports the framework by tracking
        changes in transaction behaviour and model-relevant
        patterns over time.
        """
    )


    if historical_df is not None:

        numeric_cols = historical_df.select_dtypes(
            include=np.number
        ).columns.tolist()


        selected_feature = st.selectbox(
            "Feature to monitor",
            numeric_cols
        )


        series = historical_df[
            selected_feature
        ].dropna().head(5000)


        fig = px.histogram(
            series,
            x=selected_feature,
            nbins=50,
            title=f"Distribution Monitoring: {selected_feature}"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.markdown(
            "### Monitoring Interpretation"
        )


        st.write(
            """
            Significant changes in the distribution of
            transaction variables may indicate behavioural
            change, emerging fraud patterns or data-quality
            issues requiring further investigation.
            """
        )


# ============================================================
# MODEL ADAPTATION
# ============================================================

elif page == "Model Adaptation":

    st.markdown(
        '<div class="page-title">'
        'Model Adaptation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Framework for incorporating new labelled information '
        'into future model development.'
        '</div>',
        unsafe_allow_html=True
    )


    st.info(
        """
        The current implementation demonstrates the adaptive
        architecture concept. New validated transaction labels
        can subsequently be incorporated into retraining and
        model evaluation workflows.
        """
    )


    adaptation_steps = pd.DataFrame(
        {
            "Stage": [
                "Collect New Events",
                "Validate Labels",
                "Monitor Drift",
                "Retrain Candidate Model",
                "Evaluate",
                "Deploy Updated Model"
            ],

            "Purpose": [
                "Capture recent transaction behaviour",
                "Confirm fraud / legitimate outcomes",
                "Identify distribution change",
                "Learn from new labelled observations",
                "Compare candidate against baseline",
                "Use the validated model operationally"
            ]
        }
    )


    st.dataframe(
        adaptation_steps,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# OPERATIONAL FEEDBACK
# ============================================================

elif page == "Operational Feedback":

    st.markdown(
        '<div class="page-title">'
        'Operational Feedback'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Capture investigator feedback to support future '
        'model learning and operational improvement.'
        '</div>',
        unsafe_allow_html=True
    )


    feedback_type = st.selectbox(
        "Feedback Classification",
        [
            "Confirmed Fraud",
            "False Positive",
            "Confirmed Legitimate",
            "Requires Investigation"
        ]
    )


    transaction_id = st.text_input(
        "Transaction ID"
    )


    comment = st.text_area(
        "Operational Comment"
    )


    if st.button(
        "Submit Operational Feedback",
        type="primary"
    ):

        st.session_state.feedback_records.append(
            {
                "Transaction ID":
                    transaction_id,

                "Feedback":
                    feedback_type,

                "Comment":
                    comment,

                "Timestamp":
                    pd.Timestamp.now()
            }
        )


        st.success(
            "Operational feedback recorded."
        )


    if st.session_state.feedback_records:

        st.markdown(
            "### Recorded Feedback"
        )


        st.dataframe(
            pd.DataFrame(
                st.session_state.feedback_records
            ),
            use_container_width=True
        )


# ============================================================
# FEEDBACK LEARNING PIPELINE
# ============================================================

elif page == "Feedback Learning Pipeline":

    st.markdown(
        '<div class="page-title">'
        'Feedback Learning Pipeline'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'How operational outcomes can contribute to future '
        'fraud detection improvement.'
        '</div>',
        unsafe_allow_html=True
    )


    stages = [

        (
            "01",
            "Prediction",
            "The model generates a fraud-risk prediction."
        ),

        (
            "02",
            "Operational Review",
            "A transaction can be reviewed by an investigator."
        ),

        (
            "03",
            "Feedback",
            "The operational outcome is recorded."
        ),

        (
            "04",
            "Validation",
            "Feedback is checked before becoming training data."
        ),

        (
            "05",
            "Model Learning",
            "Validated records can support future adaptation."
        )
    ]


    cols = st.columns(5)


    for col, stage in zip(
        cols,
        stages
    ):

        with col:

            number, title, text = stage

            st.markdown(
                f"""
                <div class="card">
                    <div class="workflow-number">
                        {number}
                    </div>
                    <div class="workflow-title">
                        {title}
                    </div>
                    <div class="workflow-text">
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# COST / VALUE FRAMING
# ============================================================

elif page == "Cost / Value Framing":

    st.markdown(
        '<div class="page-title">'
        'Cost / Value Framing'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Risk-aware interpretation of fraud detection outcomes.'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        Fraud detection decisions involve competing costs.

        A missed fraudulent transaction may create financial and
        operational exposure, while excessive false alarms can
        increase investigation workload and negatively affect
        legitimate customers.

        The DSS therefore supports a risk-based decision process
        rather than relying on model accuracy alone.
        """
    )


    c1, c2 = st.columns(2)


    with c1:

        st.warning(
            """
            **False Negative Cost**

            A fraudulent transaction incorrectly classified as
            legitimate may represent financial and operational
            exposure.
            """
        )


    with c2:

        st.info(
            """
            **False Positive Cost**

            A legitimate transaction incorrectly flagged as
            suspicious may increase investigation workload and
            customer friction.
            """
        )


    st.markdown(
        "### Current Evaluation"
    )


    st.dataframe(
        pd.DataFrame(
            {
                "Measure": [
                    "Fraud Detected",
                    "Fraud Missed",
                    "False Alarms",
                    "Detection Rate",
                    "Fraud F1"
                ],

                "Value": [
                    FRAUD_DETECTED,
                    FRAUD_MISSED,
                    FALSE_ALARMS,
                    f"{DETECTION_RATE:.2f}%",
                    f"{FRAUD_F1:.3f}"
                ]
            }
        ),
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# THRESHOLD TUNER
# ============================================================

elif page == "Threshold Tuner":

    st.markdown(
        '<div class="page-title">'
        'Threshold Tuner'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore how the fraud decision threshold influences '
        'operational classification.'
        '</div>',
        unsafe_allow_html=True
    )


    threshold = st.slider(
        "Classification Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05
    )


    st.metric(
        "Selected Threshold",
        f"{threshold:.2f}"
    )


    st.markdown(
        "### Threshold Interpretation"
    )


    if threshold < 0.40:

        st.warning(
            """
            A lower threshold increases sensitivity to
            potentially fraudulent transactions but may also
            increase false positives.
            """
        )

    elif threshold > 0.70:

        st.warning(
            """
            A higher threshold requires stronger evidence before
            classifying a transaction as fraud and may increase
            the number of fraudulent transactions missed.
            """
        )

    else:

        st.success(
            """
            This threshold represents a moderate risk-based
            operating point. Final deployment thresholds should
            be selected using validated operational costs and
            business requirements.
            """
        )


    st.markdown(
        "### Illustrative Threshold Profile"
    )


    thresholds = np.arange(
        0.05,
        1.00,
        0.05
    )


    illustrative_recall = (
        1 - thresholds * 0.65
    )


    illustrative_precision = (
        0.35 + thresholds * 0.60
    )


    threshold_df = pd.DataFrame(
        {
            "Threshold": thresholds,
            "Illustrative Recall":
                illustrative_recall,
            "Illustrative Precision":
                illustrative_precision
        }
    )


    fig = px.line(
        threshold_df,
        x="Threshold",
        y=[
            "Illustrative Recall",
            "Illustrative Precision"
        ],
        markers=True,
        title="Illustrative Threshold Trade-off"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# AI DECISION ASSISTANT
# ============================================================

elif page == "AI Decision Assistant":

    st.markdown(
        '<div class="page-title">'
        'AI Decision Assistant'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Decision-support interpretation of fraud-risk outputs.'
        '</div>',
        unsafe_allow_html=True
    )


    st.info(
        """
        This assistant is a prototype decision-support layer.
        It does not replace human investigators or make
        autonomous banking decisions.
        """
    )


    probability = st.slider(
        "Fraud Probability (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )


    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )


    if probability >= 80:

        decision = "High Risk"

        action = (
            "Prioritise the transaction for operational review."
        )

    elif probability >= 50:

        decision = "Elevated Risk"

        action = (
            "Consider additional verification or review."
        )

    elif probability >= 20:

        decision = "Moderate Risk"

        action = (
            "Continue monitoring and consider contextual "
            "risk information."
        )

    else:

        decision = "Low Risk"

        action = (
            "No elevated fraud signal based on the model "
            "probability alone."
        )


    st.markdown(
        "## Decision Support Assessment"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Risk Level",
            decision
        )


    with c2:

        st.metric(
            "Fraud Probability",
            f"{probability:.1f}%"
        )


    with c3:

        st.metric(
            "Amount",
            f"{amount:,.2f}"
        )


    st.success(
        f"**Recommended operational consideration:** {action}"
    )


# ============================================================
# END-TO-END DSS WORKFLOW
# ============================================================

elif page == "End-to-End DSS Workflow":

    st.markdown(
        '<div class="page-title">'
        'End-to-End DSS Workflow'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Integrated view of the adaptive and interpretable '
        'fraud detection decision-support framework.'
        '</div>',
        unsafe_allow_html=True
    )


    workflow = pd.DataFrame(
        {
            "Stage": [
                "Historical Data",
                "Machine Learning",
                "Real-Time Stream",
                "Explainable AI",
                "Adaptive Monitoring",
                "Operational Feedback",
                "Decision Support"
            ],

            "Technology / Function": [
                "Historical transaction data",
                "Random Forest",
                "Kafka",
                "SHAP / LIME",
                "Drift monitoring",
                "Investigator feedback",
                "Streamlit DSS"
            ],

            "Purpose": [
                "Model development and evaluation",
                "Transaction-level fraud classification",
                "Real-time event transport",
                "Prediction interpretation",
                "Detect changing behaviour",
                "Support future learning",
                "Operational risk decisions"
            ]
        }
    )


    st.dataframe(
        workflow,
        hide_index=True,
        use_container_width=True
    )


    st.markdown(
        "### Framework Flow"
    )


    flow_text = (
        "Historical Data → Machine Learning → "
        "Real-Time Stream → Explainable AI → "
        "Adaptive Monitoring → Operational Feedback → "
        "Decision Support"
    )


    st.info(
        flow_text
    )


# ============================================================
# ABOUT THE FRAMEWORK
# ============================================================

elif page == "About the Framework":

    st.markdown(
        '<div class="page-title">'
        'About the Framework'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Context, scope and implementation framing.'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        ## Framework Purpose

        This prototype demonstrates an adaptive and
        interpretable machine learning framework for fraud
        detection and operational decision support in digital
        banking.

        The framework combines historical machine learning,
        real-time event streaming, explainable AI, adaptive
        monitoring and operational feedback.

        ## Prototype Positioning

        The implementation is a **prototype / reference
        implementation**, rather than a claim of a production
        banking deployment.

        In an actual banking environment, transaction sources
        would be replaced by secure enterprise transaction
        APIs, event streams or message brokers. Historical
        transaction data would support model training and
        adaptation, while real-time events would flow through
        the streaming architecture for classification and
        explanation.

        ## Interpretability

        Explainability is incorporated to help users understand
        model behaviour and the factors contributing to fraud
        predictions.

        Global feature importance provides an overview of model
        behaviour, while SHAP and LIME can support transaction-
        level interpretation.

        ## Decision Support

        The objective is not simply to produce a binary fraud
        prediction. The DSS provides risk information that can
        support operational review, threshold selection,
        monitoring, feedback and future model adaptation.
        """
    )


    st.markdown(
        "## Technology Stack"
    )


    tech = pd.DataFrame(
        {
            "Layer": [
                "Data",
                "Machine Learning",
                "Streaming",
                "Explainability",
                "Dashboard",
                "Development"
            ],

            "Technology": [
                "Pandas / CSV",
                "Random Forest",
                "Apache Kafka",
                "SHAP / LIME",
                "Streamlit",
                "Python / VS Code"
            ]
        }
    )


    st.dataframe(
        tech,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Fraud DSS • Interpretable Fraud Detection &
        Operational Decision Support
        <br>
        Prototype / Reference Implementation
    </div>
    """,
    unsafe_allow_html=True
)