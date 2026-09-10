# ============================================================
# FRAUD DSS
# Interpretable Fraud Detection & Operational Decision Support
# Developed by Adeseye Samuel Ademola
# Prototype / Reference Implementation
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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "rf_fraud_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "creditcard.csv"
)

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "transaction-events"

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
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background: #f5f7fb;
    }

    .main {
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
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

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff;
    }

    .sidebar-brand {
        font-size: 24px;
        font-weight: 850;
        color: #ffffff;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        line-height: 1.5;
        color: #b8c8dc !important;
        margin-bottom: 4px;
    }

    .sidebar-author {
        font-size: 11px;
        color: #8fa7c2 !important;
        margin-bottom: 22px;
    }

    .sidebar-section {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.2px;
        color: #7794b3 !important;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .sidebar-description {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px;
        font-size: 11px;
        line-height: 1.55;
        color: #c7d5e5 !important;
        margin-top: 15px;
    }

    /* ======================================================
       HEADER
       ====================================================== */

    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
        margin-bottom: 22px;
    }

    .dashboard-title {
        font-size: 31px;
        font-weight: 850;
        color: #10233f;
        margin-bottom: 4px;
        letter-spacing: -0.6px;
    }

    .dashboard-subtitle {
        color: #687386;
        font-size: 14px;
        line-height: 1.5;
    }

    .live-badge {
        display: inline-block;
        background: #e8f8ef;
        color: #15803d;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 800;
        border: 1px solid #b9e8c9;
        white-space: nowrap;
    }

    .offline-badge {
        display: inline-block;
        background: #fff1f2;
        color: #be123c;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 800;
        border: 1px solid #fecdd3;
        white-space: nowrap;
    }

    .info-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 800;
        border: 1px solid #bfdbfe;
        white-space: nowrap;
    }

    /* ======================================================
       KPI CARDS
       ====================================================== */

    .kpi-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #e6ebf2;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.045);
        min-height: 128px;
        margin-bottom: 8px;
    }

    .kpi-icon {
        float: right;
        font-size: 21px;
        line-height: 1;
    }

    .kpi-label {
        font-size: 11px;
        color: #687386;
        font-weight: 700;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    .kpi-value {
        font-size: 27px;
        font-weight: 850;
        color: #12233f;
        margin-bottom: 5px;
    }

    .kpi-description {
        font-size: 10px;
        color: #8b95a7;
        line-height: 1.4;
    }

    /* ======================================================
       PANELS
       ====================================================== */

    .dashboard-panel {
        background: #ffffff;
        border: 1px solid #e6ebf2;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
        margin-bottom: 12px;
    }

    .panel-title {
        font-size: 15px;
        font-weight: 800;
        color: #172b4d;
        margin-bottom: 5px;
    }

    .panel-subtitle {
        font-size: 11px;
        color: #8a94a6;
        line-height: 1.5;
    }

    /* ======================================================
       SECTION TITLES
       ====================================================== */

    .section-title {
        font-size: 18px;
        font-weight: 850;
        color: #14243d;
        margin-top: 24px;
        margin-bottom: 12px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 13px;
        line-height: 1.55;
        margin-bottom: 15px;
    }

    /* ======================================================
       STATUS ROWS
       ====================================================== */

    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 15px;
        padding: 11px 0;
        border-bottom: 1px solid #edf0f5;
        font-size: 13px;
        color: #344054;
    }

    .status-row:last-child {
        border-bottom: none;
    }

    .status-good {
        color: #15803d;
        font-weight: 800;
    }

    .status-warning {
        color: #c2410c;
        font-weight: 800;
    }

    .status-danger {
        color: #dc2626;
        font-weight: 800;
    }

    .status-neutral {
        color: #64748b;
        font-weight: 700;
    }

    /* ======================================================
       INSIGHT CARDS
       ====================================================== */

    .insight-card {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e6ebf2;
        padding: 19px;
        min-height: 185px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
    }

    .insight-icon {
        font-size: 21px;
        margin-bottom: 8px;
    }

    .insight-title {
        font-size: 15px;
        font-weight: 850;
        color: #14243d;
        margin-bottom: 8px;
    }

    .insight-text {
        font-size: 13px;
        line-height: 1.7;
        color: #667085;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero-panel {
        background: linear-gradient(
            135deg,
            #0a2342 0%,
            #123b68 100%
        );
        border-radius: 18px;
        padding: 32px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(7, 27, 51, 0.16);
    }

    .hero-title {
        font-size: 28px;
        font-weight: 850;
        margin-bottom: 8px;
    }

    .hero-text {
        font-size: 14px;
        line-height: 1.7;
        color: #d9e5f2;
        max-width: 850px;
    }

    /* ======================================================
       RISK ALERTS
       ====================================================== */

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

    .unknown-alert {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #64748b;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .risk-title {
        font-size: 19px;
        font-weight: 850;
        margin-bottom: 5px;
    }

    .risk-probability {
        font-size: 30px;
        font-weight: 850;
        margin: 5px 0;
    }

    /* ======================================================
       WORKFLOW
       ====================================================== */

    .workflow-step {
        background: #ffffff;
        border: 1px solid #e6ebf2;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .workflow-number {
        display: inline-block;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        border-radius: 50%;
        background: #eaf2fb;
        color: #123b68;
        font-weight: 850;
        margin-right: 8px;
    }

    .workflow-title {
        display: inline;
        font-size: 14px;
        font-weight: 800;
        color: #172b4d;
    }

    .workflow-text {
        font-size: 12px;
        color: #687386;
        line-height: 1.6;
        margin-top: 8px;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        margin-top: 35px;
        padding: 20px 0;
        text-align: center;
        color: #98a2b3;
        font-size: 11px;
    }

    /* ======================================================
       STREAMLIT WIDGETS
       ====================================================== */

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e6ebf2;
        padding: 15px;
        border-radius: 12px;
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

        if hasattr(model, "feature_names_in_"):

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

        probability = float(
            model.predict_proba(X)[0][1]
        )

        prediction = (
            "FRAUD"
            if probability >= st.session_state.threshold
            else "LEGITIMATE"
        )

        transaction["prediction"] = prediction

        transaction["fraud_probability"] = (
            probability * 100
        )

    except Exception as error:

        transaction["prediction"] = "ERROR"

        transaction["fraud_probability"] = 0.0

        transaction["_prediction_error"] = str(
            error
        )

    return transaction


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

        group_id="fraud-dashboard-consumer-v3",

        request_timeout_ms=3000,

        session_timeout_ms=6000,

        heartbeat_interval_ms=2000
    )

    return consumer


def reset_kafka_consumer():

    try:
        create_kafka_consumer.clear()
    except Exception:
        pass


def read_kafka_transactions():

    try:

        consumer = create_kafka_consumer()

        st.session_state.kafka_status = (
            "Connected"
        )

        messages = consumer.poll(
            timeout_ms=KAFKA_POLL_TIMEOUT,
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
                    or transaction.get("id")
                    or transaction.get("Time")
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
                    in st.session_state.realtime_ids
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

        if len(
            st.session_state.realtime_ids
        ) > MAX_REALTIME_TRANSACTIONS * 2:

            st.session_state.realtime_ids = {
                str(
                    x.get("_transaction_id")
                )
                for x in
                st.session_state.realtime_transactions
                if x.get(
                    "_transaction_id"
                ) is not None
            }

        if new_transactions:

            st.session_state.last_transaction = (
                new_transactions[-1]
            )

        return new_transactions

    except Exception:

        st.session_state.kafka_status = (
            "Offline"
        )

        reset_kafka_consumer()

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
        height=320,
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
        height=320,
        margin=dict(
            l=10,
            r=10,
            t=20,
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
        height=330,
        margin=dict(
            l=10,
            r=10,
            t=20,
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
        '<div class="sidebar-section">SYSTEM NAVIGATION</div>',
        unsafe_allow_html=True
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

    st.markdown(
        """
        <div class="sidebar-description">
            <b>Prototype / Reference Implementation</b>
            <br><br>
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
        <div class="hero-panel">

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

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Welcome to the Fraud DSS</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-icon">🧠</div>

                <div class="insight-title">
                    Interpretable Machine Learning
                </div>

                <div class="insight-text">
                    Uses a Random Forest classification model
                    alongside feature importance and explainability
                    concepts to support transparent interpretation
                    of fraud predictions.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-icon">⚡</div>

                <div class="insight-title">
                    Real-Time Detection
                </div>

                <div class="insight-text">
                    Kafka provides the event-streaming layer through
                    which transaction events can be consumed and
                    evaluated by the fraud detection model.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-icon">📊</div>

                <div class="insight-title">
                    Operational Decision Support
                </div>

                <div class="insight-text">
                    Model outputs are presented as risk information
                    to support investigation, monitoring and
                    operational decision-making.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="section-title">
            Framework Objective
        </div>

        <div class="dashboard-panel">

            <div class="insight-text">
                The objective of the framework is to demonstrate
                how adaptive and interpretable machine learning
                can be integrated with real-time transaction
                ingestion and operational decision support.
                <br><br>
                It is presented as a prototype / reference
                implementation rather than a production banking
                deployment.
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
                    Fraud Detection & Operational Decision
                    Support System
                </div>
            </div>

            <div class="live-badge">
                ● SYSTEM READY
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-icon">📊</div>

                <div class="kpi-label">
                    Transactions
                </div>

                <div class="kpi-value">
                    {HISTORICAL_TRANSACTIONS:,}
                </div>

                <div class="kpi-description">
                    Historical transactions analysed
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-icon">🚨</div>

                <div class="kpi-label">
                    Fraud Cases
                </div>

                <div class="kpi-value">
                    {HISTORICAL_FRAUD:,}
                </div>

                <div class="kpi-description">
                    Identified fraudulent transactions
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-icon">📈</div>

                <div class="kpi-label">
                    Fraud Rate
                </div>

                <div class="kpi-value">
                    {HISTORICAL_FRAUD_RATE:.2f}%
                </div>

                <div class="kpi-description">
                    Fraud proportion in historical data
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-icon">🎯</div>

                <div class="kpi-label">
                    ROC-AUC
                </div>

                <div class="kpi-value">
                    {MODEL_ROC_AUC:.2f}%
                </div>

                <div class="kpi-description">
                    Model discrimination performance
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-icon">⚖️</div>

                <div class="kpi-label">
                    PR-AUC
                </div>

                <div class="kpi-value">
                    {MODEL_PR_AUC:.2f}%
                </div>

                <div class="kpi-description">
                    Fraud-class performance
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # RISK + PERFORMANCE
    # --------------------------------------------------------

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
                    Evaluation metrics from the Random
                    Forest model
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

    # --------------------------------------------------------
    # EVALUATION + PIPELINE
    # --------------------------------------------------------

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
                    <span class="status-neutral">
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

    # --------------------------------------------------------
    # EXECUTIVE INSIGHTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Executive Insights</div>',
        unsafe_allow_html=True
    )

    i1, i2, i3 = st.columns(3)

    with i1:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-icon">🚨</div>

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

                <div class="insight-icon">🎯</div>

                <div class="insight-title">
                    Detection Capability
                </div>

                <div class="insight-text">
                    The Random Forest model achieved a
                    ROC-AUC of <b>{MODEL_ROC_AUC:.2f}%</b>
                    and PR-AUC of
                    <b>{MODEL_PR_AUC:.2f}%</b>.
                    These metrics provide evidence of the
                    model's ability to discriminate between
                    fraudulent and legitimate activity.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with i3:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-icon">⚖️</div>

                <div class="insight-title">
                    Operational Consideration
                </div>

                <div class="insight-text">
                    Fraud recall is currently
                    <b>{MODEL_RECALL:.1f}%</b>, while
                    precision is
                    <b>{MODEL_PRECISION:.1f}%</b>.
                    This supports a risk-based review
                    process rather than treating every
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
                    Historical transaction behaviour,
                    fraud patterns and model performance.
                </div>
            </div>

            <div class="info-badge">
                ● HISTORICAL DATA
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    h1, h2, h3, h4 = st.columns(4)

    h1.metric(
        "Transactions",
        f"{HISTORICAL_TRANSACTIONS:,}"
    )

    h2.metric(
        "Fraud Cases",
        f"{HISTORICAL_FRAUD:,}"
    )

    h3.metric(
        "Fraud Rate",
        f"{HISTORICAL_FRAUD_RATE:.2f}%"
    )

    h4.metric(
        "ROC-AUC",
        f"{MODEL_ROC_AUC:.2f}%"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            "### Transaction Risk Distribution"
        )

        st.plotly_chart(
            create_donut_chart(
                HISTORICAL_FRAUD,
                HISTORICAL_LEGITIMATE
            ),
            width="stretch"
        )

    with c2:

        st.markdown(
            "### Model Evaluation"
        )

        st.plotly_chart(
            create_model_performance_chart(),
            width="stretch"
        )

    amount_chart = create_amount_chart()

    if amount_chart is not None:

        st.markdown(
            "### Transaction Amount Distribution"
        )

        st.plotly_chart(
            amount_chart,
            width="stretch"
        )

    if historical_df is not None:

        st.markdown(
            "### Historical Dataset Preview"
        )

        st.dataframe(
            historical_df.head(20),
            width="stretch",
            height=400
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
                    Submit transaction characteristics to obtain
                    a model-generated fraud risk assessment.
                </div>
            </div>

            <div class="info-badge">
                ● MODEL INFERENCE
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None:

        st.error(
            "Random Forest model could not be loaded."
        )

    else:

        st.markdown(
            """
            <div class="dashboard-panel">

                <div class="panel-title">
                    Transaction Input
                </div>

                <div class="panel-subtitle">
                    Enter the transaction amount and optional
                    model features. Unspecified PCA features
                    default to zero.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            input_time = st.number_input(
                "Transaction Time",
                value=0.0
            )

        with col2:

            input_amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=100.0
            )

        transaction_input = {
            "Time": input_time,
            "Amount": input_amount
        }

        with st.expander(
            "Advanced Model Features (V1–V28)"
        ):

            feature_cols = st.columns(4)

            for i in range(1, 29):

                with feature_cols[
                    (i - 1) % 4
                ]:

                    transaction_input[
                        f"V{i}"
                    ] = st.number_input(
                        f"V{i}",
                        value=0.0,
                        key=f"predict_v{i}"
                    )

        if st.button(
            "🔍 Analyse Transaction",
            type="primary"
        ):

            result = predict_transaction(
                transaction_input
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
                            🚨 FRAUD RISK DETECTED
                        </div>

                        <div>
                            Estimated fraud probability
                        </div>

                        <div class="risk-probability">
                            {probability:.2f}%
                        </div>

                        <div>
                            Risk-based review is recommended.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="legitimate-alert">

                        <div class="risk-title">
                            ✓ LOW FRAUD RISK
                        </div>

                        <div>
                            Estimated fraud probability
                        </div>

                        <div class="risk-probability">
                            {probability:.2f}%
                        </div>

                        <div>
                            The transaction is below the
                            configured classification threshold.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
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
                    classification from the Kafka event stream.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    read_kafka_transactions()

    if (
        st.session_state.kafka_status
        == "Connected"
    ):

        st.markdown(
            """
            <span class="live-badge">
                ● KAFKA STREAM ACTIVE
            </span>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <span class="offline-badge">
                ● KAFKA OFFLINE — RETRYING
            </span>
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

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Transactions Received",
        received_count
    )

    k2.metric(
        "Fraud Detected",
        fraud_count
    )

    k3.metric(
        "Legitimate",
        legitimate_count
    )

    k4.metric(
        "Real-Time Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    latest_probability = 0

    if st.session_state.last_transaction:

        latest_probability = (
            st.session_state.last_transaction
            .get(
                "fraud_probability",
                0
            )
        )

    k5.metric(
        "Latest Risk",
        f"{latest_probability:.2f}%"
    )

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

            else:

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

        else:

            st.info(
                "No transaction has been received yet."
            )

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
        "Kafka is consumed only while this page is active."
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
                    Explore historical transactions and
                    inspect available transaction attributes.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if historical_df is None:

        st.warning(
            "creditcard.csv could not be loaded."
        )

    else:

        search_amount = st.number_input(
            "Minimum Transaction Amount",
            min_value=0.0,
            value=0.0
        )

        filtered = historical_df[
            historical_df["Amount"]
            >= search_amount
        ]

        st.write(
            f"Showing {len(filtered):,} transactions."
        )

        st.dataframe(
            filtered.head(500),
            width="stretch",
            height=500
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
                    risk-related patterns.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if historical_df is not None:

        if "Time" in historical_df.columns:

            trend = (
                historical_df
                .assign(
                    Time_Bin=lambda x:
                    pd.cut(
                        x["Time"],
                        bins=20
                    )
                )
                .groupby(
                    "Time_Bin",
                    observed=False
                )
                .agg(
                    Transactions=("Class", "size"),
                    Fraud=("Class", "sum")
                )
                .reset_index()
            )

            fig = px.line(
                trend,
                x="Time_Bin",
                y=[
                    "Transactions",
                    "Fraud"
                ],
                markers=True
            )

            fig.update_layout(
                height=400
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        else:

            st.info(
                "Time information is not available."
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
                    Monitoring changes in transaction behaviour
                    and model operating conditions.
                </div>
            </div>

            <div class="info-badge">
                ● ADAPTIVE LAYER
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Current Session Events",
        len(
            st.session_state.realtime_transactions
        )
    )

    c2.metric(
        "Configured Threshold",
        f"{st.session_state.threshold:.0%}"
    )

    c3.metric(
        "Model Recall",
        f"{MODEL_RECALL:.2f}%"
    )

    st.markdown(
        """
        <div class="dashboard-panel">

            <div class="panel-title">
                Adaptive Monitoring Concept
            </div>

            <div class="insight-text">
                The framework is designed to monitor changes
                in incoming transaction behaviour, model
                performance and operational feedback.
                Significant changes can provide evidence for
                further model evaluation or adaptation.
            </div>

        </div>
        """,
        unsafe_allow_html=True
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
                    Framework for monitoring when model
                    retraining or adaptation may be required.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    steps = [
        (
            "1",
            "Monitor",
            "Observe incoming transactions and performance indicators."
        ),
        (
            "2",
            "Detect Change",
            "Identify potential shifts in transaction or fraud patterns."
        ),
        (
            "3",
            "Validate",
            "Assess whether the observed change is persistent and meaningful."
        ),
        (
            "4",
            "Adapt",
            "Retrain or update the model using appropriate validated data."
        ),
        (
            "5",
            "Evaluate",
            "Compare the adapted model against the established baseline."
        )
    ]

    for number, title, text in steps:

        st.markdown(
            f"""
            <div class="workflow-step">

                <span class="workflow-number">
                    {number}
                </span>

                <span class="workflow-title">
                    {title}
                </span>

                <div class="workflow-text">
                    {text}
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
                    Capture human review outcomes to support
                    continuous learning and evaluation.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.last_transaction:

        latest = st.session_state.last_transaction

        st.write(
            "Latest monitored transaction:"
        )

        st.json(
            latest
        )

    feedback = st.selectbox(
        "Operational outcome",
        [
            "No feedback recorded",
            "Confirmed Fraud",
            "False Positive",
            "Confirmed Legitimate",
            "Requires Further Investigation"
        ]
    )

    if st.button(
        "Save Operational Feedback"
    ):

        st.session_state.feedback_records.append(
            {
                "feedback": feedback,
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }
        )

        st.success(
            "Operational feedback recorded."
        )

    if st.session_state.feedback_records:

        st.dataframe(
            pd.DataFrame(
                st.session_state.feedback_records
            ),
            width="stretch",
            hide_index=True
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
                    Connecting operational feedback with
                    future model evaluation and adaptation.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    steps = [
        "Transaction prediction",
        "Operational review",
        "Feedback capture",
        "Label validation",
        "Historical feedback repository",
        "Model evaluation",
        "Potential retraining",
        "Validation and deployment decision"
    ]

    for i, step in enumerate(
        steps,
        start=1
    ):

        st.markdown(
            f"""
            <div class="workflow-step">

                <span class="workflow-number">
                    {i}
                </span>

                <span class="workflow-title">
                    {step}
                </span>

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
                    Understanding fraud detection as an
                    operational risk and decision problem.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-icon">
                    💰
                </div>

                <div class="insight-title">
                    Fraud Loss
                </div>

                <div class="insight-text">
                    Missed fraudulent transactions can create
                    direct financial and operational losses.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-icon">
                    🔎
                </div>

                <div class="insight-title">
                    Investigation Cost
                </div>

                <div class="insight-text">
                    Excessive false positives can increase
                    manual review requirements and operational
                    workload.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-icon">
                    ⚖️
                </div>

                <div class="insight-title">
                    Decision Trade-off
                </div>

                <div class="insight-text">
                    Threshold selection should balance missed
                    fraud against unnecessary intervention.
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
                    Explore how the fraud classification threshold
                    affects operational sensitivity.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    threshold = st.slider(
        "Fraud Classification Threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(
            st.session_state.threshold
        ),
        step=0.05
    )

    st.session_state.threshold = threshold

    st.metric(
        "Current Threshold",
        f"{threshold:.0%}"
    )

    st.info(
        "A lower threshold increases sensitivity to potential "
        "fraud, while a higher threshold requires stronger "
        "model evidence before classifying an event as fraud."
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
                    Structured decision-support guidance based
                    on model risk information and operational context.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    latest = st.session_state.last_transaction

    if latest is None:

        st.info(
            "No real-time transaction is currently available. "
            "Use Real-Time Fraud Detection first."
        )

    else:

        probability = latest.get(
            "fraud_probability",
            0
        )

        prediction = latest.get(
            "prediction",
            "UNKNOWN"
        )

        st.markdown(
            f"""
            <div class="dashboard-panel">

                <div class="panel-title">
                    Current Risk Assessment
                </div>

                <div class="status-row">
                    <span>Prediction</span>
                    <span>
                        {prediction}
                    </span>
                </div>

                <div class="status-row">
                    <span>Fraud Probability</span>
                    <span>
                        {probability:.2f}%
                    </span>
                </div>

                <div class="status-row">
                    <span>Threshold</span>
                    <span>
                        {st.session_state.threshold:.0%}
                    </span>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction == "FRAUD":

            st.warning(
                "Decision-support recommendation: "
                "prioritise the transaction for risk-based review "
                "and consider additional verification."
            )

        else:

            st.success(
                "Decision-support recommendation: "
                "no elevated fraud intervention is indicated "
                "by the current model threshold."
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
            "1",
            "Historical Data",
            "Historical transaction data supports analysis and model development."
        ),
        (
            "2",
            "Preprocessing",
            "Transaction variables are prepared and scaled where required."
        ),
        (
            "3",
            "Machine Learning",
            "Random Forest produces fraud classifications and probabilities."
        ),
        (
            "4",
            "Real-Time Ingestion",
            "Kafka provides the event-streaming layer for transaction events."
        ),
        (
            "5",
            "Risk Classification",
            "Incoming events are processed and evaluated by the model."
        ),
        (
            "6",
            "Explainability",
            "Feature importance and XAI methods support interpretation."
        ),
        (
            "7",
            "Decision Support",
            "Risk information is presented for operational review."
        ),
        (
            "8",
            "Feedback",
            "Operational outcomes can contribute to future model evaluation."
        ),
        (
            "9",
            "Adaptation",
            "Validated changes can inform model monitoring and retraining."
        )
    ]

    for number, title, text in workflow:

        st.markdown(
            f"""
            <div class="workflow-step">

                <span class="workflow-number">
                    {number}
                </span>

                <span class="workflow-title">
                    {title}
                </span>

                <div class="workflow-text">
                    {text}
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
                    & Operational Decision Support
                </div>
            </div>

            <div class="info-badge">
                ● PROTOTYPE
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-panel">

            <div class="hero-title">
                Fraud Detection Decision Support System
            </div>

            <div class="hero-text">
                This prototype demonstrates an adaptive and
                interpretable machine learning framework for
                fraud detection and operational decision support.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div class="dashboard-panel">

                <div class="panel-title">
                    Framework Characteristics
                </div>

                <div class="status-row">
                    <span>Machine Learning</span>
                    <span class="status-good">
                        Random Forest
                    </span>
                </div>

                <div class="status-row">
                    <span>Real-Time Streaming</span>
                    <span class="status-good">
                        Kafka
                    </span>
                </div>

                <div class="status-row">
                    <span>Dashboard</span>
                    <span class="status-good">
                        Streamlit
                    </span>
                </div>

                <div class="status-row">
                    <span>Explainability</span>
                    <span class="status-good">
                        XAI / Feature Importance
                    </span>
                </div>

                <div class="status-row">
                    <span>Decision Support</span>
                    <span class="status-good">
                        Risk-Based
                    </span>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="dashboard-panel">

                <div class="panel-title">
                    Implementation Position
                </div>

                <div class="insight-text">
                    The system is a prototype / reference
                    implementation intended to demonstrate
                    the architecture and decision-support
                    concepts.
                    <br><br>
                    In an actual banking deployment, enterprise
                    transaction sources, secure APIs or event
                    streaming infrastructure would replace the
                    synthetic transaction generation used for
                    demonstration.
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
        Fraud DSS • Interpretable Fraud Detection &
        Operational Decision Support
        <br>
        Prototype / Reference Implementation
        • Developed by Adeseye Samuel Ademola
    </div>
    """,
    unsafe_allow_html=True
)