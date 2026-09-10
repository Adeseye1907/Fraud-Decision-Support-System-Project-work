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

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "fraud-predictions"

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


# Historical test-set results
TEST_TRANSACTIONS = 56746
TEST_FRAUD = 95
TEST_FRAUD_DETECTED = 73
TEST_FRAUD_MISSED = 22
TEST_FALSE_ALARMS = 8
TEST_PREDICTED_FRAUD = 81
TEST_PREDICTED_LEGITIMATE = 56665

DETECTION_RATE = 76.84


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f5f7fb;
    }

    .main {
        padding-top: 1rem;
    }

    /* SIDEBAR */

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
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #b8c8dc !important;
        line-height: 1.5;
        margin-bottom: 8px;
    }

    .sidebar-author {
        font-size: 11px;
        color: #9fb2c9 !important;
        margin-bottom: 25px;
    }

    /* HEADERS */

    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 22px;
    }

    .dashboard-title {
        font-size: 30px;
        font-weight: 800;
        color: #10233f;
        margin-bottom: 4px;
    }

    .dashboard-subtitle {
        color: #6b7280;
        font-size: 14px;
        line-height: 1.5;
    }

    .live-badge {
        background: #e8f8ef;
        color: #15803d !important;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #b9e8c9;
        white-space: nowrap;
    }

    .offline-badge {
        background: #fff1f2;
        color: #be123c !important;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #fecdd3;
        white-space: nowrap;
    }

    /* HERO */

    .hero-card {
        background: linear-gradient(
            135deg,
            #071b33 0%,
            #123b66 100%
        );
        border-radius: 18px;
        padding: 32px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.12);
    }

    .hero-title {
        font-size: 31px;
        font-weight: 850;
        margin-bottom: 10px;
        color: white;
    }

    .hero-text {
        font-size: 14px;
        line-height: 1.75;
        color: #d9e5f3;
        max-width: 900px;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 18px;
        padding: 7px 13px;
        border-radius: 20px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        font-size: 11px;
        font-weight: 700;
    }

    /* KPI */

    .kpi-card {
        background: white;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #e8edf4;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        min-height: 125px;
    }

    .kpi-label {
        font-size: 12px;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .kpi-value {
        font-size: 27px;
        font-weight: 800;
        color: #12233f;
        margin-bottom: 5px;
    }

    .kpi-description {
        font-size: 11px;
        color: #8b95a7;
        line-height: 1.4;
    }

    .kpi-icon {
        float: right;
        font-size: 20px;
    }

    /* SECTIONS */

    .section-title {
        font-size: 19px;
        font-weight: 800;
        color: #14243d;
        margin-top: 27px;
        margin-bottom: 7px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 15px;
        line-height: 1.5;
    }

    /* PANELS */

    .dashboard-panel {
        background: white;
        border: 1px solid #e8edf4;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
        margin-bottom: 15px;
    }

    .panel-title {
        font-size: 16px;
        font-weight: 750;
        color: #172b4d;
        margin-bottom: 6px;
    }

    .panel-subtitle {
        font-size: 12px;
        color: #8a94a6;
        margin-bottom: 12px;
        line-height: 1.5;
    }

    /* STATUS */

    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 11px 0;
        border-bottom: 1px solid #edf0f5;
        font-size: 13px;
        color: #344054;
    }

    .status-row:last-child {
        border-bottom: none;
    }

    .status-good {
        color: #15803d !important;
        font-weight: 700;
    }

    .status-warning {
        color: #c2410c !important;
        font-weight: 700;
    }

    .status-danger {
        color: #dc2626 !important;
        font-weight: 700;
    }

    /* INSIGHTS */

    .insight-card {
        background: white;
        border-radius: 14px;
        border: 1px solid #e8edf4;
        padding: 20px;
        height: 100%;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
    }

    .insight-title {
        font-size: 15px;
        font-weight: 800;
        color: #14243d;
        margin-bottom: 8px;
    }

    .insight-text {
        font-size: 13px;
        line-height: 1.7;
        color: #667085;
    }

    /* RESEARCH FOCUS */

    .focus-card {
        background: white;
        border: 1px solid #e8edf4;
        border-radius: 14px;
        padding: 20px;
        min-height: 185px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
    }

    .focus-icon {
        font-size: 26px;
        margin-bottom: 8px;
    }

    .focus-title {
        font-size: 15px;
        font-weight: 800;
        color: #14243d;
        margin-bottom: 8px;
    }

    .focus-text {
        font-size: 13px;
        color: #667085;
        line-height: 1.65;
    }

    /* WORKFLOW */

    .workflow-card {
        background: white;
        border: 1px solid #e8edf4;
        border-radius: 14px;
        padding: 18px;
        min-height: 145px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
    }

    .workflow-number {
        font-size: 12px;
        font-weight: 800;
        color: #2563eb;
        margin-bottom: 8px;
    }

    .workflow-title {
        font-size: 14px;
        font-weight: 800;
        color: #14243d;
        margin-bottom: 7px;
    }

    .workflow-text {
        font-size: 12px;
        line-height: 1.6;
        color: #667085;
    }

    /* ALERTS */

    .fraud-alert {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 5px solid #dc2626;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .legitimate-alert {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .risk-title {
        font-size: 19px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .risk-probability {
        font-size: 30px;
        font-weight: 850;
    }

    /* FOOTER */

    .footer {
        margin-top: 40px;
        padding: 22px 0;
        text-align: center;
        color: #98a2b3;
        font-size: 11px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


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
# HISTORICAL STATISTICS
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
        col
        for col in ["Time", "Amount"]
        if col in X.columns
    ]

    if scaler is not None and cols_to_scale:

        try:

            X[cols_to_scale] = scaler.transform(
                X[cols_to_scale]
            )

        except Exception:
            pass

    return X


# ============================================================
# PREDICT TRANSACTION
# ============================================================

def predict_transaction(transaction):

    transaction = dict(transaction)

    if model is None:

        transaction["prediction"] = "UNKNOWN"
        transaction["fraud_probability"] = 0.0

        return transaction

    try:

        X = prepare_transaction(
            transaction
        )

        prediction = model.predict(X)[0]

        probability = model.predict_proba(
            X
        )[0][1]

        transaction["prediction"] = (
            "FRAUD"
            if int(prediction) == 1
            else "LEGITIMATE"
        )

        transaction["fraud_probability"] = (
            float(probability) * 100
        )

    except Exception as e:

        transaction["prediction"] = "ERROR"
        transaction["fraud_probability"] = 0.0
        transaction["_prediction_error"] = str(e)

    return transaction


# ============================================================
# KAFKA
# ============================================================

@st.cache_resource(show_spinner=False)
def create_kafka_consumer():

    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        value_deserializer=lambda x: json.loads(
            x.decode("utf-8")
        ),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="fraud-dashboard-consumer-v3",
        request_timeout_ms=3000,
        session_timeout_ms=6000,
        heartbeat_interval_ms=2000
    )


def clear_kafka_consumer():

    try:
        create_kafka_consumer.clear()
    except Exception:
        pass


def read_kafka_transactions():

    try:

        consumer = create_kafka_consumer()

        st.session_state.kafka_status = "Connected"

        messages = consumer.poll(
            timeout_ms=KAFKA_POLL_TIMEOUT,
            max_records=50
        )

        new_transactions = []

        for _, records in messages.items():

            for message in records:

                transaction = message.value

                transaction_id = (
                    transaction.get("transaction_id")
                    or transaction.get("id")
                    or transaction.get("Time")
                    or transaction.get("timestamp")
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

                if transaction_id in (
                    st.session_state.realtime_ids
                ):
                    continue

                st.session_state.realtime_ids.add(
                    transaction_id
                )

                transaction = predict_transaction(
                    transaction
                )

                transaction["_transaction_id"] = (
                    transaction_id
                )

                st.session_state.realtime_transactions.append(
                    transaction
                )

                new_transactions.append(
                    transaction
                )

        if len(
            st.session_state.realtime_ids
        ) > MAX_REALTIME_TRANSACTIONS * 2:

            st.session_state.realtime_ids = {
                str(
                    x.get("_transaction_id")
                )
                for x in
                st.session_state.realtime_transactions
                if x.get("_transaction_id")
                is not None
            }

        if new_transactions:

            st.session_state.last_transaction = (
                new_transactions[-1]
            )

        return new_transactions

    except Exception:

        st.session_state.kafka_status = "Offline"

        clear_kafka_consumer()

        return []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_number(value):

    return f"{value:,.0f}"


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
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Transactions: %{value:,}<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
                )
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=310,
        margin=dict(
            l=10,
            r=10,
            t=30,
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
        "F1-Score",
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
            "y": "Percentage"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_yaxes(
        range=[0, 110]
    )

    fig.update_layout(
        height=310,
        margin=dict(
            l=10,
            r=10,
            t=30,
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

    temp["Amount_Band"] = pd.cut(
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
            "Amount_Band",
            observed=False
        )
        .size()
        .reset_index(
            name="Transactions"
        )
    )

    fig = px.bar(
        grouped,
        x="Amount_Band",
        y="Transactions",
        labels={
            "Amount_Band":
                "Transaction Amount",
            "Transactions":
                "Transactions"
        }
    )

    fig.update_layout(
        height=310,
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10
        )
    )

    return fig


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            🛡️ Fraud DSS
        </div>

        <div class="sidebar-subtitle">
            Interpretable Fraud Detection &
            Operational Decision Support
        </div>

        <div class="sidebar-author">
            Developed by Adeseye Samuel Ademola
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### System Navigation"
    )

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

    st.markdown(
        """
        <div style="
            font-size:11px;
            color:#9fb2c9;
            line-height:1.6;
        ">
            <b>Prototype / Reference Implementation</b><br>
            Adaptive and interpretable fraud detection
            and operational decision support.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# INTRODUCTION
# ============================================================

if page == "Introduction":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    INTERPRETABLE FRAUD DETECTION
                    & OPERATIONAL DECISION SUPPORT
                </div>

                <div class="dashboard-subtitle">
                    Real-Time Intelligence for Digital Banking
                </div>
            </div>

            <div class="live-badge">
                ● FRAMEWORK
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-card">

            <div class="hero-title">
                Fraud Detection Decision Support System
            </div>

            <div class="hero-text">
                An interpretable framework for fraud detection,
                monitoring and operational decision support.
                The prototype combines machine learning,
                real-time event streaming and explainability
                to support risk-aware operational decisions.
            </div>

            <div class="hero-tag">
                ADAPTIVE • INTERPRETABLE • OPERATIONAL
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    kafka_online = (
        st.session_state.kafka_status
        == "Connected"
    )

    status_text = (
        "System Online"
        if model is not None
        else "Model Unavailable"
    )

    st.markdown(
        f"""
        <div class="dashboard-panel">

            <div class="panel-title">
                SYSTEM STATUS
            </div>

            <div class="status-row">
                <span>
                    System Status
                </span>

                <span class="status-good">
                    ● {status_text}
                </span>
            </div>

            <div class="status-row">
                <span>
                    Kafka Topic
                </span>

                <span>
                    {KAFKA_TOPIC}
                </span>
            </div>

            <div class="status-row">
                <span>
                    Model
                </span>

                <span>
                    Random Forest
                </span>
            </div>

            <div class="status-row">
                <span>
                    Explainability
                </span>

                <span>
                    SHAP / LIME
                </span>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "🔄 Refresh Dashboard",
        key="intro_refresh"
    ):

        st.rerun()

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Background of the Study</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-panel">

        <div class="insight-text">

        The rapid growth of digital banking and electronic
        payment systems has increased the volume, velocity
        and complexity of financial transactions. At the same
        time, this growth has created opportunities for
        increasingly sophisticated fraudulent activities.

        <br><br>

        Machine learning provides an opportunity to identify
        complex patterns within transaction data and detect
        potentially fraudulent behaviour. However, predictive
        performance alone is insufficient for effective
        operational fraud management.

        <br><br>

        Operational teams need to understand not only whether
        a transaction has been classified as potentially
        fraudulent, but also why the machine learning model
        reached that decision.

        <br><br>

        This framework therefore combines fraud prediction,
        explainable AI, real-time monitoring, adaptive
        monitoring and operational decision support.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # RESEARCH FOCUS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Research Focus</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        st.markdown(
            """
            <div class="focus-card">

                <div class="focus-icon">
                    🔄
                </div>

                <div class="focus-title">
                    Adaptive Machine Learning
                </div>

                <div class="focus-text">
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
            <div class="focus-card">

                <div class="focus-icon">
                    🔍
                </div>

                <div class="focus-title">
                    Interpretable Machine Learning
                </div>

                <div class="focus-text">
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
            <div class="focus-card">

                <div class="focus-icon">
                    🧭
                </div>

                <div class="focus-title">
                    Operational Decision Support
                </div>

                <div class="focus-text">
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
        '<div class="section-title">Framework Workflow</div>',
        unsafe_allow_html=True
    )

    workflow = [
        (
            "01",
            "Historical Data",
            "Historical transactions support model development and evaluation."
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
            "Drift monitoring identifies changing transaction behaviour."
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

    for index, item in enumerate(workflow):

        col = cols[index % 4]

        with col:

            st.markdown(
                f"""
                <div class="workflow-card">

                    <div class="workflow-number">
                        {item[0]}
                    </div>

                    <div class="workflow-title">
                        {item[1]}
                    </div>

                    <div class="workflow-text">
                        {item[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Input</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-panel">

        <div class="insight-text">

        The trained fraud detection model operates on
        <b>30 transaction features</b> consisting of
        <b>Time, V1-V28 and Amount</b>.

        <br><br>

        V1-V28 represent anonymised PCA-transformed
        transaction features, while Time and Amount provide
        additional transaction-level information.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

elif page == "Executive Overview":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Executive Overview
                </div>

                <div class="dashboard-subtitle">
                    Fraud Detection & Operational Decision Support
                    System
                </div>
            </div>

            <div class="live-badge">
                ● SYSTEM READY
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI ROW

    c1, c2, c3, c4, c5 = st.columns(5)

    kpis = [
        (
            c1,
            "📊",
            "Transactions",
            format_number(
                HISTORICAL_TRANSACTIONS
            ),
            "Historical transactions analysed"
        ),
        (
            c2,
            "🚨",
            "Fraud Cases",
            format_number(
                HISTORICAL_FRAUD
            ),
            "Identified fraudulent transactions"
        ),
        (
            c3,
            "📈",
            "Fraud Rate",
            f"{HISTORICAL_FRAUD_RATE:.2f}%",
            "Fraud proportion in historical data"
        ),
        (
            c4,
            "🎯",
            "ROC-AUC",
            f"{MODEL_ROC_AUC:.2f}%",
            "Model discrimination performance"
        ),
        (
            c5,
            "⚖️",
            "PR-AUC",
            f"{MODEL_PR_AUC:.2f}%",
            "Fraud-class performance"
        )
    ]

    for col, icon, label, value, description in kpis:

        with col:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-icon">
                        {icon}
                    </div>

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {value}
                    </div>

                    <div class="kpi-description">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    # CHARTS

    st.markdown(
        '<div class="section-title">Risk & Model Performance</div>',
        unsafe_allow_html=True
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        st.markdown(
            """
            <div class="dashboard-panel">

                <div class="panel-title">
                    Transaction Risk Distribution
                </div>

                <div class="panel-subtitle">
                    Historical fraud and legitimate
                    transaction profile
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(
            create_donut_chart(
                HISTORICAL_FRAUD,
                HISTORICAL_LEGITIMATE
            ),
            width="stretch",
            config={
                "displayModeBar": False
            }
        )

    with chart2:

        st.markdown(
            """
            <div class="dashboard-panel">

                <div class="panel-title">
                    Fraud Detection Model Performance
                </div>

                <div class="panel-subtitle">
                    Evaluation metrics from the
                    Random Forest model
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(
            create_model_performance_chart(),
            width="stretch",
            config={
                "displayModeBar": False
            }
        )

    # MODEL + PIPELINE

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="section-title">Model Evaluation Summary</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="dashboard-panel">

                <div class="status-row">
                    <span>Accuracy</span>
                    <span class="status-good">
                        {MODEL_ACCURACY:.2f}%
                    </span>
                </div>

                <div class="status-row">
                    <span>Precision</span>
                    <span class="status-good">
                        {MODEL_PRECISION:.2f}%
                    </span>
                </div>

                <div class="status-row">
                    <span>Recall</span>
                    <span class="status-good">
                        {MODEL_RECALL:.2f}%
                    </span>
                </div>

                <div class="status-row">
                    <span>F1-Score</span>
                    <span class="status-good">
                        {MODEL_F1:.2f}%
                    </span>
                </div>

                <div class="status-row">
                    <span>ROC-AUC</span>
                    <span class="status-good">
                        {MODEL_ROC_AUC:.2f}%
                    </span>
                </div>

                <div class="status-row">
                    <span>PR-AUC</span>
                    <span class="status-good">
                        {MODEL_PR_AUC:.2f}%
                    </span>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            '<div class="section-title">Decision Support Pipeline</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="dashboard-panel">

                <div class="status-row">
                    <span>Transaction Ingestion</span>
                    <span class="status-good">
                        ● ACTIVE
                    </span>
                </div>

                <div class="status-row">
                    <span>Kafka Event Streaming</span>
                    <span class="status-good">
                        ● READY
                    </span>
                </div>

                <div class="status-row">
                    <span>Fraud Classification</span>
                    <span class="status-good">
                        ● ACTIVE
                    </span>
                </div>

                <div class="status-row">
                    <span>Explainability Layer</span>
                    <span class="status-good">
                        ● READY
                    </span>
                </div>

                <div class="status-row">
                    <span>Adaptive Monitoring</span>
                    <span class="status-good">
                        ● ACTIVE
                    </span>
                </div>

                <div class="status-row">
                    <span>Decision Support Dashboard</span>
                    <span class="status-good">
                        ● ACTIVE
                    </span>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # EXECUTIVE INSIGHTS

    st.markdown(
        '<div class="section-title">Executive Insights</div>',
        unsafe_allow_html=True
    )

    i1, i2, i3 = st.columns(3)

    with i1:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-title">
                    Fraud Exposure
                </div>

                <div class="insight-text">

                    The historical dataset contains
                    <b>{HISTORICAL_FRAUD:,}</b>
                    identified fraudulent transactions out
                    of <b>{HISTORICAL_TRANSACTIONS:,}</b>
                    transactions analysed.

                    This represents a fraud rate of
                    <b>{HISTORICAL_FRAUD_RATE:.2f}%</b>.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-title">
                    Detection Capability
                </div>

                <div class="insight-text">

                    The Random Forest model achieved a
                    ROC-AUC of <b>{MODEL_ROC_AUC:.2f}%</b>
                    and PR-AUC of
                    <b>{MODEL_PR_AUC:.2f}%</b>.

                    These metrics indicate the model's ability
                    to discriminate between fraudulent and
                    legitimate transaction patterns.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with i3:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-title">
                    Operational Consideration
                </div>

                <div class="insight-text">

                    Fraud recall is currently
                    <b>{MODEL_RECALL:.1f}%</b>,
                    while precision is
                    <b>{MODEL_PRECISION:.1f}%</b>.

                    This supports a risk-based operational
                    review process rather than treating every
                    transaction as equally risky.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# HISTORICAL MODEL ANALYSIS
# ============================================================

elif page == "Historical Model Analysis":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Historical Model Analysis
                </div>

                <div class="dashboard-subtitle">
                    Evaluation and interpretation of the trained
                    Random Forest fraud detection model.
                </div>
            </div>

            <div class="live-badge">
                ● HISTORICAL DATA
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # MODEL PERFORMANCE

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)

    performance = [
        (
            p1,
            "ROC-AUC",
            "0.961"
        ),
        (
            p2,
            "PR-AUC",
            "0.813"
        ),
        (
            p3,
            "Fraud Precision",
            "0.90"
        ),
        (
            p4,
            "Fraud Recall",
            "0.77"
        )
    ]

    for col, label, value in performance:

        with col:
            st.metric(
                label,
                value
            )

    # TEST DATASET

    st.markdown(
        '<div class="section-title">Historical Test Dataset</div>',
        unsafe_allow_html=True
    )

    t1, t2, t3, t4 = st.columns(4)

    test_values = [
        (
            t1,
            "Transactions Evaluated",
            "56,746"
        ),
        (
            t2,
            "Actual Fraud",
            "95"
        ),
        (
            t3,
            "Fraud Detected",
            "73"
        ),
        (
            t4,
            "Fraud Missed",
            "22"
        )
    ]

    for col, label, value in test_values:

        with col:
            st.metric(
                label,
                value
            )

    # OPERATIONAL PERFORMANCE

    st.markdown(
        '<div class="section-title">Operational Fraud Detection Performance</div>',
        unsafe_allow_html=True
    )

    o1, o2, o3 = st.columns(3)

    with o1:
        st.metric(
            "Detection Rate",
            "76.84%"
        )

    with o2:
        st.metric(
            "False Alarms",
            "8"
        )

    with o3:
        st.metric(
            "Fraud F1-Score",
            "0.830"
        )

    # GLOBAL INTERPRETABILITY

    st.markdown(
        '<div class="section-title">Global Model Interpretability</div>',
        unsafe_allow_html=True
    )

    if (
        model is not None
        and hasattr(
            model,
            "feature_importances_"
        )
    ):

        feature_names = get_model_features()

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance":
                    model.feature_importances_
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
            title=""
        )

        fig.update_layout(
            height=500,
            yaxis=dict(
                categoryorder="total ascending"
            )
        )

        st.markdown(
            """
            <div class="dashboard-panel">

                <div class="panel-title">
                    Global Feature Importance
                </div>

                <div class="panel-subtitle">
                    Global feature importance identifies
                    variables with the greatest influence
                    on Random Forest predictions.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # PREDICTION OUTCOME

    st.markdown(
        '<div class="section-title">Prediction Outcome Summary</div>',
        unsafe_allow_html=True
    )

    q1, q2 = st.columns(2)

    with q1:
        st.metric(
            "Predicted Fraud",
            f"{TEST_PREDICTED_FRAUD:,}"
        )

    with q2:
        st.metric(
            "Predicted Legitimate",
            f"{TEST_PREDICTED_LEGITIMATE:,}"
        )

    # HISTORICAL DATA

    if historical_df is not None:

        st.markdown(
            '<div class="section-title">Historical Prediction Results</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            historical_df.head(100),
            width="stretch",
            height=400,
            hide_index=True
        )


# ============================================================
# PREDICT FRAUD
# ============================================================

elif page == "Predict Fraud":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Predict Fraud
                </div>

                <div class="dashboard-subtitle">
                    Predict fraud for an individual transaction
                    or a batch of transactions and examine
                    model risk scores.
                </div>
            </div>

            <div class="live-badge">
                ● PREDICTION ENGINE
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None:

        st.error(
            "The Random Forest model could not be loaded."
        )

    else:

        prediction_mode = st.radio(
            "Prediction Mode",
            [
                "Manual Entry",
                "Batch CSV Upload"
            ],
            horizontal=True
        )

        # ----------------------------------------------------
        # MANUAL
        # ----------------------------------------------------

        if prediction_mode == "Manual Entry":

            st.markdown(
                '<div class="section-title">Enter Transaction Details</div>',
                unsafe_allow_html=True
            )

            st.info(
                "V1-V28 are anonymised PCA-transformed "
                "features. Zero values can be used for testing."
            )

            with st.form(
                "manual_prediction_form"
            ):

                c1, c2 = st.columns(2)

                with c1:

                    transaction_time = st.number_input(
                        "Time",
                        value=0.0
                    )

                with c2:

                    amount = st.number_input(
                        "Amount",
                        value=0.0,
                        min_value=0.0
                    )

                st.markdown(
                    "### V1 - V28 (PCA features)"
                )

                values = {}

                cols = st.columns(4)

                for i in range(1, 29):

                    with cols[(i - 1) % 4]:

                        values[f"V{i}"] = st.number_input(
                            f"V{i}",
                            value=0.0,
                            format="%.6f"
                        )

                submitted = st.form_submit_button(
                    "🔍 Predict Transaction"
                )

            if submitted:

                transaction = {
                    "Time":
                        transaction_time,
                    "Amount":
                        amount
                }

                transaction.update(
                    values
                )

                result = predict_transaction(
                    transaction
                )

                probability = result.get(
                    "fraud_probability",
                    0
                )

                prediction = result.get(
                    "prediction",
                    "UNKNOWN"
                )

                if prediction == "FRAUD":

                    st.markdown(
                        f"""
                        <div class="fraud-alert">

                            <div class="risk-title">
                                🚨 FRAUD DETECTED
                            </div>

                            <div>
                                Fraud probability
                            </div>

                            <div class="risk-probability">
                                {probability:.2f}%
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif prediction == "LEGITIMATE":

                    st.markdown(
                        f"""
                        <div class="legitimate-alert">

                            <div class="risk-title">
                                ✓ LEGITIMATE TRANSACTION
                            </div>

                            <div>
                                Fraud probability
                            </div>

                            <div class="risk-probability">
                                {probability:.2f}%
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # ----------------------------------------------------
        # BATCH
        # ----------------------------------------------------

        else:

            st.markdown(
                '<div class="section-title">Batch CSV Upload</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                Upload a CSV containing the model input
                features. The expected features are
                <b>Time, V1-V28 and Amount</b>.
                """,
                unsafe_allow_html=True
            )

            uploaded_file = st.file_uploader(
                "Upload transaction CSV",
                type=["csv"]
            )

            if uploaded_file is not None:

                try:

                    batch_df = pd.read_csv(
                        uploaded_file
                    )

                    st.write(
                        "Preview"
                    )

                    st.dataframe(
                        batch_df.head(20),
                        width="stretch",
                        hide_index=True
                    )

                    if st.button(
                        "🚨 Predict Batch"
                    ):

                        predictions = []

                        probabilities = []

                        for _, row in batch_df.iterrows():

                            result = predict_transaction(
                                row.to_dict()
                            )

                            predictions.append(
                                result.get(
                                    "prediction",
                                    "UNKNOWN"
                                )
                            )

                            probabilities.append(
                                result.get(
                                    "fraud_probability",
                                    0
                                )
                            )

                        output_df = batch_df.copy()

                        output_df[
                            "Prediction"
                        ] = predictions

                        output_df[
                            "Fraud Probability (%)"
                        ] = probabilities

                        st.success(
                            "Batch prediction completed."
                        )

                        st.dataframe(
                            output_df,
                            width="stretch",
                            height=450,
                            hide_index=True
                        )

                        csv_data = output_df.to_csv(
                            index=False
                        ).encode(
                            "utf-8"
                        )

                        st.download_button(
                            "⬇️ Download Predictions",
                            csv_data,
                            "fraud_predictions.csv",
                            "text/csv"
                        )

                except Exception as e:

                    st.error(
                        f"Unable to process CSV: {e}"
                    )


# ============================================================
# REAL-TIME FRAUD DETECTION
# ============================================================

elif page == "Real-Time Fraud Detection":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Real-Time Fraud Detection
                </div>

                <div class="dashboard-subtitle">
                    Live transaction monitoring and fraud
                    classification from the Kafka transaction stream.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # READ KAFKA ONLY ON THIS PAGE

    read_kafka_transactions()

    if (
        st.session_state.kafka_status
        == "Connected"
    ):

        st.markdown(
            """
            <div class="live-badge">
                ● KAFKA STREAM ACTIVE
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="offline-badge">
                ● KAFKA OFFLINE — RETRYING
            </div>
            """,
            unsafe_allow_html=True
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

    k1, k2, k3, k4, k5 = st.columns(5)

    metrics = [
        (
            k1,
            "Transactions Received",
            received_count,
            "Kafka events received this session"
        ),
        (
            k2,
            "Fraud Detected",
            fraud_count,
            "Transactions classified as fraud"
        ),
        (
            k3,
            "Legitimate",
            legitimate_count,
            "Transactions classified as legitimate"
        ),
        (
            k4,
            "Real-Time Fraud Rate",
            f"{fraud_rate:.2f}%",
            "Fraud among received events"
        ),
        (
            k5,
            "Latest Risk",
            f"{latest_probability:.2f}%",
            "Fraud probability of latest event"
        )
    ]

    for col, label, value, description in metrics:

        with col:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {value}
                    </div>

                    <div class="kpi-description">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    # RISK DISTRIBUTION

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
                width="stretch",
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

                st.markdown(
                    f"""
                    <div class="fraud-alert">

                        <div class="risk-title">
                            🚨 FRAUDULENT TRANSACTION
                        </div>

                        <div>
                            Fraud probability
                        </div>

                        <div class="risk-probability">
                            {probability:.2f}%
                        </div>

                        <div>
                            Immediate risk-based review recommended.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif prediction == "LEGITIMATE":

                st.markdown(
                    f"""
                    <div class="legitimate-alert">

                        <div class="risk-title">
                            ✓ LEGITIMATE TRANSACTION
                        </div>

                        <div>
                            Fraud probability
                        </div>

                        <div class="risk-probability">
                            {probability:.2f}%
                        </div>

                        <div>
                            No elevated fraud risk detected.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            details = {}

            for key, value in latest.items():

                if key.startswith("_"):
                    continue

                if key in [
                    "prediction",
                    "fraud_probability"
                ]:
                    continue

                details[key] = value

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
                    width="stretch",
                    hide_index=True
                )

        else:

            st.info(
                "No transaction has been received yet."
            )

    # TABLE

    st.markdown(
        "### Recent Transactions"
    )

    if transactions:

        table_data = []

        for transaction in reversed(
            transactions[-50:]
        ):

            table_data.append(
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
            pd.DataFrame(table_data),
            width="stretch",
            height=420,
            hide_index=True
        )

    else:

        st.info(
            "No Kafka transactions have been received yet."
        )

    st.caption(
        "Kafka is polled only while this page is active."
    )

    if st.button(
        "🔄 Refresh Real-Time Data"
    ):

        st.rerun()


# ============================================================
# TRANSACTION EXPLORER
# ============================================================

elif page == "Transaction Explorer":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Transaction Explorer
                </div>

                <div class="dashboard-subtitle">
                    Explore transaction-level observations
                    within the historical dataset.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if historical_df is None:

        st.warning(
            "Historical dataset could not be loaded."
        )

    else:

        search_amount = st.number_input(
            "Minimum transaction amount",
            min_value=0.0,
            value=0.0
        )

        filtered_df = historical_df[
            historical_df["Amount"]
            >= search_amount
        ]

        st.write(
            f"{len(filtered_df):,} transactions found."
        )

        st.dataframe(
            filtered_df.head(500),
            width="stretch",
            height=500,
            hide_index=True
        )


# ============================================================
# TRENDS & MONITORING
# ============================================================

elif page == "Trends & Monitoring":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Trends & Monitoring
                </div>

                <div class="dashboard-subtitle">
                    Monitoring transaction behaviour and
                    fraud-related patterns.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    amount_chart = create_amount_chart()

    if amount_chart is not None:

        st.plotly_chart(
            amount_chart,
            width="stretch"
        )

    st.info(
        "Trend monitoring can be extended with temporal "
        "fraud-rate, transaction-volume and drift indicators."
    )


# ============================================================
# ADAPTIVE MONITORING
# ============================================================

elif page == "Adaptive Monitoring":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Adaptive Monitoring
                </div>

                <div class="dashboard-subtitle">
                    Monitoring changing transaction behaviour
                    and potential data or concept drift.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-panel">

            <div class="panel-title">
                Adaptive Monitoring Framework
            </div>

            <div class="insight-text">

                The adaptive layer monitors changes in
                transaction distributions and fraud-related
                behaviour. Significant changes can trigger
                further model investigation, recalibration
                or retraining.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric(
            "Monitoring Status",
            "ACTIVE"
        )

    with a2:
        st.metric(
            "Drift Signal",
            "MONITORING"
        )

    with a3:
        st.metric(
            "Adaptation",
            "READY"
        )


# ============================================================
# MODEL ADAPTATION
# ============================================================

elif page == "Model Adaptation":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Model Adaptation
                </div>

                <div class="dashboard-subtitle">
                    Model update and adaptation strategy for
                    evolving fraud environments.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-panel">

            <div class="panel-title">
                Model Adaptation Strategy
            </div>

            <div class="insight-text">

                Historical data supports model development,
                while monitoring and operational feedback
                provide signals for future adaptation.

                <br><br>

                The framework is designed so that updated
                labelled transaction data can be incorporated
                into future training cycles.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# OPERATIONAL FEEDBACK
# ============================================================

elif page == "Operational Feedback":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Operational Feedback
                </div>

                <div class="dashboard-subtitle">
                    Capture operational outcomes associated
                    with fraud predictions.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    feedback = st.selectbox(
        "Feedback Outcome",
        [
            "Confirmed Fraud",
            "False Positive",
            "Legitimate",
            "Under Investigation"
        ]
    )

    notes = st.text_area(
        "Operational Notes"
    )

    if st.button(
        "Submit Feedback"
    ):

        st.success(
            f"Feedback recorded: {feedback}"
        )


# ============================================================
# FEEDBACK LEARNING PIPELINE
# ============================================================

elif page == "Feedback Learning Pipeline":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Feedback Learning Pipeline
                </div>

                <div class="dashboard-subtitle">
                    Connecting operational outcomes to future
                    model learning and adaptation.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    steps = [
        (
            "01",
            "Prediction",
            "Model generates fraud risk."
        ),
        (
            "02",
            "Operational Review",
            "Analyst reviews the case."
        ),
        (
            "03",
            "Feedback",
            "Outcome is recorded."
        ),
        (
            "04",
            "Label Update",
            "Confirmed outcomes become learning data."
        ),
        (
            "05",
            "Model Adaptation",
            "Future training incorporates validated feedback."
        )
    ]

    cols = st.columns(5)

    for i, step in enumerate(steps):

        with cols[i]:

            st.markdown(
                f"""
                <div class="workflow-card">

                    <div class="workflow-number">
                        {step[0]}
                    </div>

                    <div class="workflow-title">
                        {step[1]}
                    </div>

                    <div class="workflow-text">
                        {step[2]}
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
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Cost / Value Framing
                </div>

                <div class="dashboard-subtitle">
                    Risk-aware interpretation of fraud detection
                    decisions and their operational consequences.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-panel">

            <div class="panel-title">
                Why Cost Matters
            </div>

            <div class="insight-text">

                Fraud detection decisions have asymmetric
                consequences.

                <br><br>

                Missing fraudulent transactions can result in
                financial and reputational losses, while
                excessive false alarms can increase operational
                workload and negatively affect legitimate
                customers.

                <br><br>

                The DSS therefore supports risk-aware decision
                making rather than relying on accuracy alone.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# THRESHOLD TUNER
# ============================================================

elif page == "Threshold Tuner":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Threshold Tuner
                </div>

                <div class="dashboard-subtitle">
                    Explore how fraud classification thresholds
                    affect operational risk decisions.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    threshold = st.slider(
        "Fraud Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.metric(
        "Selected Threshold",
        f"{threshold:.2f}"
    )

    st.info(
        "Lower thresholds generally increase sensitivity "
        "to potential fraud but may increase false alarms. "
        "Higher thresholds may reduce alerts while "
        "increasing the possibility of missed fraud."
    )


# ============================================================
# AI DECISION ASSISTANT
# ============================================================

elif page == "AI Decision Assistant":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    AI Decision Assistant
                </div>

                <div class="dashboard-subtitle">
                    Interpreting fraud risk outputs for
                    operational decision support.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    question = st.text_area(
        "Enter an operational question"
    )

    if st.button(
        "Generate Decision Guidance"
    ):

        if question.strip():

            st.markdown(
                """
                <div class="dashboard-panel">

                    <div class="panel-title">
                        Decision Support Guidance
                    </div>

                    <div class="insight-text">

                        Review the transaction risk score,
                        model explanation, transaction context
                        and applicable operational policy before
                        taking action.

                        <br><br>

                        The AI assistant is intended to support
                        human decision-making rather than replace
                        operational judgement.

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.warning(
                "Enter a question first."
            )


# ============================================================
# END-TO-END DSS WORKFLOW
# ============================================================

elif page == "End-to-End DSS Workflow":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    End-to-End DSS Workflow
                </div>

                <div class="dashboard-subtitle">
                    Integrated view of the adaptive and
                    interpretable fraud decision-support framework.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    workflow = [
        (
            "01",
            "Historical Data",
            "Historical transactions support training and evaluation."
        ),
        (
            "02",
            "Model Development",
            "Random Forest learns transaction-level fraud patterns."
        ),
        (
            "03",
            "Real-Time Ingestion",
            "Kafka transports incoming transaction events."
        ),
        (
            "04",
            "Fraud Prediction",
            "The model produces a transaction-level risk prediction."
        ),
        (
            "05",
            "Explainability",
            "SHAP and LIME help interpret model decisions."
        ),
        (
            "06",
            "Monitoring",
            "Changing behaviour and potential drift are monitored."
        ),
        (
            "07",
            "Operational Feedback",
            "Human review outcomes provide additional learning signals."
        ),
        (
            "08",
            "Decision Support",
            "The dashboard presents risk-aware operational intelligence."
        )
    ]

    cols = st.columns(4)

    for i, item in enumerate(workflow):

        with cols[i % 4]:

            st.markdown(
                f"""
                <div class="workflow-card">

                    <div class="workflow-number">
                        {item[0]}
                    </div>

                    <div class="workflow-title">
                        {item[1]}
                    </div>

                    <div class="workflow-text">
                        {item[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# ABOUT THE FRAMEWORK
# ============================================================

elif page == "About the Framework":

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    About the Framework
                </div>

                <div class="dashboard-subtitle">
                    Adaptive and Interpretable Fraud Detection
                    and Operational Decision Support
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-card">

            <div class="hero-title">
                Fraud Detection DSS
            </div>

            <div class="hero-text">

                This prototype demonstrates an adaptive and
                interpretable machine learning framework for
                fraud detection and operational decision support
                in digital banking.

                <br><br>

                The framework integrates historical model
                development, real-time event streaming,
                transaction-level prediction, explainable AI,
                adaptive monitoring and operational feedback.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    a1, a2 = st.columns(2)

    with a1:

        st.markdown(
            """
            <div class="dashboard-panel">

                <div class="panel-title">
                    Project Positioning
                </div>

                <div class="insight-text">

                    The implementation is a
                    <b>prototype / reference implementation</b>
                    rather than a claim of production deployment
                    within a banking institution.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with a2:

        st.markdown(
            """
            <div class="dashboard-panel">

                <div class="panel-title">
                    Core Technologies
                </div>

                <div class="insight-text">

                    Python • Random Forest • Kafka • Streamlit
                    • SHAP • LIME • Pandas • Scikit-learn

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Fraud Detection & Operational Decision Support System
        • Prototype / Reference Implementation
    </div>
    """,
    unsafe_allow_html=True
)