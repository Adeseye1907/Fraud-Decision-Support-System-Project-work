# ============================================================
# FRAUD DSS — DASHBOARD.PY
# Interpretable Fraud Detection & Operational Decision Support
# Developed by Adeseye Samuel Ademola
#
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
from kafka.errors import KafkaError


warnings.filterwarnings("ignore")


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
# PATHS / CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "rf_fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "creditcard.csv")

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
# CUSTOM CSS
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
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
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
        background: transparent;
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
        color: #b8c8dc !important;
        line-height: 1.5;
        margin-bottom: 4px;
    }

    .sidebar-author {
        font-size: 11px;
        color: #8fa6c0 !important;
        margin-bottom: 25px;
    }

    .sidebar-section {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.3px;
        color: #8fa6c0 !important;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    .sidebar-footer {
        margin-top: 30px;
        padding-top: 15px;
        border-top: 1px solid rgba(255,255,255,0.12);
        font-size: 10px;
        line-height: 1.6;
        color: #8fa6c0 !important;
    }


    /* ======================================================
       MAIN HEADER
       ====================================================== */

    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
    }

    .dashboard-title {
        font-size: 30px;
        font-weight: 800;
        color: #10233f;
        line-height: 1.2;
    }

    .dashboard-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-top: 5px;
        line-height: 1.5;
    }

    .live-badge {
        background: #e8f8ef;
        color: #15803d;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid #b9e8c9;
        white-space: nowrap;
    }

    .offline-badge {
        background: #fff1f2;
        color: #be123c;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid #fecdd3;
        white-space: nowrap;
    }


    /* ======================================================
       HERO
       ====================================================== */

    .hero-panel {
        background: linear-gradient(
            135deg,
            #0b2747 0%,
            #123b68 100%
        );
        border-radius: 18px;
        padding: 30px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.12);
    }

    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-text {
        font-size: 14px;
        line-height: 1.7;
        color: #dbe8f6;
        max-width: 900px;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 15px;
        padding: 7px 12px;
        border-radius: 20px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.15);
        font-size: 11px;
        font-weight: 700;
    }


    /* ======================================================
       KPI CARDS
       ====================================================== */

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


    /* ======================================================
       PANELS
       ====================================================== */

    .dashboard-panel {
        background: white;
        border: 1px solid #e8edf4;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
    }

    .panel-title {
        font-size: 15px;
        font-weight: 750;
        color: #172b4d;
        margin-bottom: 5px;
    }

    .panel-subtitle {
        font-size: 11px;
        color: #8a94a6;
        margin-bottom: 12px;
        line-height: 1.5;
    }


    /* ======================================================
       SECTION HEADINGS
       ====================================================== */

    .section-title {
        font-size: 18px;
        font-weight: 800;
        color: #14243d;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 15px;
    }


    /* ======================================================
       STATUS ROWS
       ====================================================== */

    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 11px 0;
        border-bottom: 1px solid #edf0f5;
        font-size: 13px;
        color: #344054;
        gap: 15px;
    }

    .status-row:last-child {
        border-bottom: none;
    }

    .status-good {
        color: #15803d;
        font-weight: 700;
    }

    .status-warning {
        color: #c2410c;
        font-weight: 700;
    }

    .status-danger {
        color: #dc2626;
        font-weight: 700;
    }


    /* ======================================================
       INSIGHT CARDS
       ====================================================== */

    .insight-card {
        background: white;
        border-radius: 14px;
        border: 1px solid #e8edf4;
        padding: 18px;
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
        line-height: 1.65;
        color: #667085;
    }


    /* ======================================================
       ALERTS
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

    .risk-title {
        font-size: 19px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .risk-probability {
        font-size: 30px;
        font-weight: 850;
    }


    /* ======================================================
       INFO BOX
       ====================================================== */

    .info-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 16px;
        color: #475467;
        font-size: 13px;
        line-height: 1.7;
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
    HISTORICAL_TRANSACTIONS - HISTORICAL_FRAUD
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
# KAFKA CONSUMER
# ============================================================

@st.cache_resource(show_spinner=False)
def create_kafka_consumer():

    consumer = KafkaConsumer(
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

    return consumer


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

                transaction_id = str(transaction_id)

                if transaction_id in st.session_state.realtime_ids:
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

        if len(st.session_state.realtime_ids) > (
            MAX_REALTIME_TRANSACTIONS * 2
        ):

            st.session_state.realtime_ids = {
                str(x.get("_transaction_id"))
                for x in st.session_state.realtime_transactions
                if x.get("_transaction_id")
            }

        if new_transactions:

            st.session_state.last_transaction = (
                new_transactions[-1]
            )

        return new_transactions

    except Exception:

        st.session_state.kafka_status = "Offline"

        try:
            create_kafka_consumer.clear()
        except Exception:
            pass

        return []


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

            return list(model.feature_names_in_)

    return DEFAULT_FEATURES


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
# PREDICTION
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
            Interpretable Fraud Detection & Operational
            Decision Support
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

    navigation = [
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
    ]

    page = st.radio(
        "System Navigation",
        navigation,
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class="sidebar-footer">
            <b>Prototype / Reference Implementation</b><br><br>

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
                    Fraud Detection Decision Support System
                </div>

                <div class="dashboard-subtitle">
                    An interpretable framework for fraud detection,
                    monitoring and operational decision support.
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
        <div class="hero-panel">

            <div class="hero-title">
                Welcome to the Fraud DSS
            </div>

            <div class="hero-text">
                This prototype demonstrates an adaptive and
                interpretable machine learning framework for
                detecting fraudulent transactions and supporting
                operational decision-making.
            </div>

            <div class="hero-tag">
                INTERPRETABLE • ADAPTIVE • OPERATIONAL
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

                <div class="insight-title">
                    Fraud Detection
                </div>

                <div class="insight-text">
                    Machine learning is used to distinguish
                    potentially fraudulent transactions from
                    legitimate activity.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-title">
                    Explainability
                </div>

                <div class="insight-text">
                    The framework supports interpretation of
                    model behaviour so that predictions can be
                    examined rather than treated as unexplained
                    outputs.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-title">
                    Decision Support
                </div>

                <div class="insight-text">
                    Detection outputs are translated into
                    operational risk information that can
                    support human review and intervention.
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
    # RISK / PERFORMANCE
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
                    Historical fraudulent versus legitimate
                    transaction profile.
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
            use_container_width=True,
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
                    Evaluation metrics from the Random Forest
                    classification model.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(
            create_model_performance_chart(),
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    # --------------------------------------------------------
    # MODEL / PIPELINE
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

        kafka_ready = (
            st.session_state.kafka_status
            == "Connected"
        )

        kafka_text = (
            "● ACTIVE"
            if kafka_ready
            else "● READY"
        )

        st.markdown(
            f"""
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
                        {kafka_text}
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
                    <span>Decision Support</span>
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

                <div class="insight-title">
                    Fraud Exposure
                </div>

                <div class="insight-text">
                    The historical dataset contains
                    <b>{HISTORICAL_FRAUD:,}</b>
                    fraudulent transactions out of
                    <b>{HISTORICAL_TRANSACTIONS:,}</b>
                    transactions analysed.

                    This corresponds to a historical fraud
                    rate of
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

                    These metrics provide a broader view of
                    model discrimination and fraud-class
                    performance.
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
                    Fraud recall is
                    <b>{MODEL_RECALL:.1f}%</b>,
                    while precision is
                    <b>{MODEL_PRECISION:.1f}%</b>.

                    The framework therefore supports
                    risk-based review rather than treating
                    every transaction identically.
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
                    Historical transaction behaviour and
                    fraud-model evaluation.
                </div>
            </div>

            <div class="live-badge">
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
            use_container_width=True
        )

    with c2:

        st.markdown(
            "### Model Evaluation"
        )

        st.plotly_chart(
            create_model_performance_chart(),
            use_container_width=True
        )

    amount_chart = create_amount_chart()

    if amount_chart is not None:

        st.markdown(
            "### Transaction Amount Distribution"
        )

        st.plotly_chart(
            amount_chart,
            use_container_width=True
        )

    if historical_df is not None:

        st.markdown(
            "### Historical Dataset Preview"
        )

        st.dataframe(
            historical_df.head(20),
            use_container_width=True,
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
                    Submit transaction characteristics for
                    model-based fraud risk assessment.
                </div>
            </div>

            <div class="live-badge">
                ● MODEL READY
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
            <div class="info-box">
                Enter the transaction features below. The model
                will return a classification and estimated fraud
                probability.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        col1, col2 = st.columns(2)

        with col1:

            amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=100.0
            )

        with col2:

            transaction_time = st.number_input(
                "Transaction Time",
                min_value=0.0,
                value=0.0
            )

        st.markdown(
            "### PCA Transaction Features"
        )

        feature_values = {}

        cols = st.columns(4)

        for i in range(1, 29):

            with cols[(i - 1) % 4]:

                feature_values[f"V{i}"] = st.number_input(
                    f"V{i}",
                    value=0.0,
                    format="%.6f",
                    key=f"predict_v{i}"
                )

        if st.button(
            "🔍 Analyse Transaction",
            use_container_width=True
        ):

            transaction = {
                "Time": transaction_time,
                "Amount": amount,
                **feature_values
            }

            result = predict_transaction(
                transaction
            )

            prediction = result.get(
                "prediction",
                "UNKNOWN"
            )

            probability = result.get(
                "fraud_probability",
                0.0
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

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="legitimate-alert">

                        <div class="risk-title">
                            ✓ LOWER FRAUD RISK
                        </div>

                        <div>
                            Estimated fraud probability
                        </div>

                        <div class="risk-probability">
                            {probability:.2f}%
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
                    Live transaction monitoring from the Kafka
                    transaction stream.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Kafka is ONLY consumed on this page.

    read_kafka_transactions()

    if st.session_state.kafka_status == "Connected":

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

    received_count = len(transactions)

    fraud_count = sum(
        1
        for x in transactions
        if x.get("prediction") == "FRAUD"
    )

    legitimate_count = sum(
        1
        for x in transactions
        if x.get("prediction") == "LEGITIMATE"
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
            st.session_state
            .last_transaction
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
                use_container_width=True
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
                            Risk-based review recommended.
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
                    use_container_width=True,
                    hide_index=True
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
            use_container_width=True,
            height=420,
            hide_index=True
        )

    else:

        st.info(
            "No Kafka transactions have been received yet."
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
                    Explore historical transaction records and
                    fraud labels.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if historical_df is None:

        st.error(
            "Historical dataset could not be loaded."
        )

    else:

        search_term = st.text_input(
            "Search transaction records"
        )

        filtered_df = historical_df

        if search_term:

            mask = filtered_df.astype(
                str
            ).apply(
                lambda row:
                    row.str.contains(
                        search_term,
                        case=False,
                        na=False
                    ).any(),
                axis=1
            )

            filtered_df = filtered_df[mask]

        st.write(
            f"Showing {len(filtered_df):,} records"
        )

        st.dataframe(
            filtered_df.head(500),
            use_container_width=True,
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
                    Monitoring transaction volumes, fraud levels
                    and transaction-value patterns.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if historical_df is None:

        st.warning(
            "Historical dataset is not available."
        )

    else:

        if "Amount" in historical_df.columns:

            fig = px.histogram(
                historical_df,
                x="Amount",
                nbins=50,
                title="Transaction Amount Distribution"
            )

            fig.update_layout(
                height=420
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if "Class" in historical_df.columns:

            class_counts = (
                historical_df["Class"]
                .value_counts()
                .rename(
                    index={
                        0: "Legitimate",
                        1: "Fraud"
                    }
                )
                .reset_index()
            )

            class_counts.columns = [
                "Classification",
                "Transactions"
            ]

            fig = px.bar(
                class_counts,
                x="Classification",
                y="Transactions",
                title="Transaction Classification"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
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
                    Monitoring the framework for changes in
                    transaction and fraud behaviour.
                </div>
            </div>

            <div class="live-badge">
                ● MONITORING
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Historical Fraud Rate",
        f"{HISTORICAL_FRAUD_RATE:.2f}%"
    )

    c2.metric(
        "Model Recall",
        f"{MODEL_RECALL:.2f}%"
    )

    c3.metric(
        "Model PR-AUC",
        f"{MODEL_PR_AUC:.2f}%"
    )

    st.markdown(
        """
        <div class="info-box">
            Adaptive monitoring provides a mechanism for
            observing changes in transaction distributions,
            fraud prevalence and model behaviour. In a production
            implementation, these signals can support model
            review, recalibration and retraining decisions.
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
                    Model lifecycle and adaptation strategy.
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
                Adaptive Model Lifecycle
            </div>

            <div class="panel-subtitle">
                Monitoring → Feedback → Evaluation →
                Adaptation → Redeployment
            </div>

            <div class="status-row">
                <span>Current Model</span>
                <span class="status-good">
                    Random Forest
                </span>
            </div>

            <div class="status-row">
                <span>Historical Training Data</span>
                <span class="status-good">
                    Available
                </span>
            </div>

            <div class="status-row">
                <span>Feedback Integration</span>
                <span class="status-good">
                    Framework Ready
                </span>
            </div>

            <div class="status-row">
                <span>Model Retraining</span>
                <span class="status-warning">
                    Human / Operational Trigger
                </span>
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
                    Capture analyst and operational decisions
                    associated with fraud alerts.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    transaction_id = st.text_input(
        "Transaction ID"
    )

    decision = st.selectbox(
        "Operational Decision",
        [
            "Confirmed Fraud",
            "False Positive",
            "Under Review",
            "Legitimate"
        ]
    )

    notes = st.text_area(
        "Analyst / Operational Notes"
    )

    if st.button(
        "Submit Feedback"
    ):

        st.success(
            "Operational feedback captured for this prototype session."
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
                    Connecting operational feedback to future
                    model improvement.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    steps = [
        "Fraud Prediction",
        "Operational Review",
        "Feedback Capture",
        "Label Validation",
        "Performance Evaluation",
        "Model Adaptation"
    ]

    for index, step in enumerate(
        steps,
        start=1
    ):

        st.markdown(
            f"""
            <div class="dashboard-panel"
                 style="margin-bottom:10px;">

                <div class="status-row">

                    <span>
                        <b>{index}.</b>
                        {step}
                    </span>

                    <span class="status-good">
                        ● FRAMEWORK STAGE
                    </span>

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
                    Risk-based interpretation of fraud detection
                    decisions.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Precision",
        f"{MODEL_PRECISION:.2f}%"
    )

    c2.metric(
        "Recall",
        f"{MODEL_RECALL:.2f}%"
    )

    c3.metric(
        "PR-AUC",
        f"{MODEL_PR_AUC:.2f}%"
    )

    st.markdown(
        """
        <div class="info-box">
            Fraud detection decisions involve a trade-off
            between detecting fraudulent activity and generating
            unnecessary alerts. A decision-support framework can
            therefore incorporate operational costs, customer
            impact, investigation effort and potential fraud loss
            when determining appropriate thresholds.
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
                    Explore how a probability threshold changes
                    the operational classification rule.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    threshold = st.slider(
        "Fraud Decision Threshold",
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
        f"""
        <div class="info-box">
            Transactions with an estimated fraud probability
            greater than or equal to <b>{threshold:.2f}</b>
            would be classified as fraud under this threshold.
            The appropriate operational threshold should be
            determined using validation data and the relative
            costs of false positives and false negatives.
        </div>
        """,
        unsafe_allow_html=True
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
                    Structured decision-support interpretation of
                    fraud risk information.
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

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            st.markdown(
                """
                <div class="dashboard-panel">

                    <div class="panel-title">
                        Decision-Support Guidance
                    </div>

                    <div class="insight-text">
                        Review the transaction's fraud
                        probability, model classification,
                        available explanation features and
                        operational context before making a
                        final decision.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
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
            "Transaction Ingestion",
            "Historical or real-time transaction events enter the framework."
        ),
        (
            "02",
            "Preprocessing",
            "Transaction features are prepared for model inference."
        ),
        (
            "03",
            "Fraud Classification",
            "The Random Forest model estimates fraud risk."
        ),
        (
            "04",
            "Explainability",
            "Model behaviour can be interpreted using feature importance and XAI techniques."
        ),
        (
            "05",
            "Operational Decision Support",
            "Risk information is presented to support human review."
        ),
        (
            "06",
            "Feedback",
            "Operational outcomes can provide information for future adaptation."
        ),
        (
            "07",
            "Model Adaptation",
            "Validated feedback and monitoring signals can inform model updates."
        )
    ]

    for number, title, description in workflow:

        st.markdown(
            f"""
            <div class="dashboard-panel"
                 style="margin-bottom:12px;">

                <div class="status-row">

                    <span>
                        <b>{number}</b>
                        &nbsp;&nbsp;
                        <b>{title}</b>
                    </span>

                    <span class="status-good">
                        DSS STAGE
                    </span>

                </div>

                <div class="panel-subtitle"
                     style="margin-top:8px;">
                    {description}
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
                    Adaptive and Interpretable Machine Learning
                    Framework for Fraud Detection and Operational
                    Decision Support.
                </div>
            </div>

            <div class="live-badge">
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
                Fraud DSS
            </div>

            <div class="hero-text">
                This system is a prototype/reference
                implementation demonstrating how machine
                learning, real-time event streaming,
                explainability, monitoring and operational
                feedback can be combined within a fraud
                decision-support framework.
            </div>

            <div class="hero-tag">
                ADAPTIVE & INTERPRETABLE
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
                    Core Technologies
                </div>

                <div class="status-row">
                    <span>Machine Learning</span>
                    <span>Random Forest</span>
                </div>

                <div class="status-row">
                    <span>Streaming</span>
                    <span>Apache Kafka</span>
                </div>

                <div class="status-row">
                    <span>Dashboard</span>
                    <span>Streamlit</span>
                </div>

                <div class="status-row">
                    <span>Explainability</span>
                    <span>Feature Importance / XAI</span>
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
                    Framework Principles
                </div>

                <div class="status-row">
                    <span>Interpretability</span>
                    <span class="status-good">
                        ✓
                    </span>
                </div>

                <div class="status-row">
                    <span>Adaptability</span>
                    <span class="status-good">
                        ✓
                    </span>
                </div>

                <div class="status-row">
                    <span>Real-Time Monitoring</span>
                    <span class="status-good">
                        ✓
                    </span>
                </div>

                <div class="status-row">
                    <span>Operational Decision Support</span>
                    <span class="status-good">
                        ✓
                    </span>
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
        <br>
        Developed by Adeseye Samuel Ademola
    </div>
    """,
    unsafe_allow_html=True
)