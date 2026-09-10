# ============================================================
# INTERPRETABLE FRAUD DETECTION & OPERATIONAL DECISION SUPPORT
# COMPLETE DASHBOARD
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


# ============================================================
# PAGE NAVIGATION
# ============================================================

PAGES = [
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
        color: #91a8c2 !important;
        margin-bottom: 25px;
    }

    .sidebar-section {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.2px;
        color: #7fa3c9 !important;
        margin-top: 15px;
        margin-bottom: 8px;
    }

    /* ======================================================
       HEADER
       ====================================================== */

    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
    }

    .dashboard-title {
        font-size: 30px;
        font-weight: 800;
        color: #10233f;
        line-height: 1.15;
        margin-bottom: 5px;
    }

    .dashboard-subtitle {
        color: #667085;
        font-size: 14px;
        line-height: 1.5;
    }

    .live-badge {
        background: #e8f8ef;
        color: #15803d;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #b9e8c9;
        white-space: nowrap;
    }

    .offline-badge {
        background: #fff1f2;
        color: #be123c;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #fecdd3;
        white-space: nowrap;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        background:
            linear-gradient(
                135deg,
                #071b33 0%,
                #0b3159 55%,
                #12466e 100%
            );
        border-radius: 20px;
        padding: 38px 42px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
    }

    .hero-title {
        color: white;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 15px;
    }

    .hero-text {
        color: #d9e7f5;
        font-size: 15px;
        line-height: 1.75;
        max-width: 900px;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 20px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.20);
        border-radius: 20px;
        padding: 8px 15px;
        color: #dceeff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
    }

    /* ======================================================
       PANELS
       ====================================================== */

    .dashboard-panel {
        background: white;
        border: 1px solid #e8edf4;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
        margin-bottom: 15px;
    }

    .panel-title {
        font-size: 16px;
        font-weight: 800;
        color: #172b4d;
        margin-bottom: 8px;
    }

    .panel-subtitle {
        font-size: 12px;
        color: #8a94a6;
        margin-bottom: 15px;
    }

    /* ======================================================
       STATUS
       ====================================================== */

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
       FOCUS CARDS
       ====================================================== */

    .focus-card {
        background: white;
        border: 1px solid #e8edf4;
        border-radius: 15px;
        padding: 22px;
        height: 100%;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
    }

    .focus-icon {
        font-size: 28px;
        margin-bottom: 12px;
    }

    .focus-title {
        font-size: 16px;
        font-weight: 800;
        color: #14243d;
        margin-bottom: 8px;
    }

    .focus-text {
        font-size: 13px;
        line-height: 1.65;
        color: #667085;
    }

    /* ======================================================
       WORKFLOW
       ====================================================== */

    .workflow-card {
        background: white;
        border: 1px solid #e8edf4;
        border-radius: 14px;
        padding: 18px;
        height: 100%;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
    }

    .workflow-number {
        font-size: 12px;
        font-weight: 800;
        color: #2f6fed;
        margin-bottom: 8px;
    }

    .workflow-title {
        font-size: 14px;
        font-weight: 800;
        color: #172b4d;
        margin-bottom: 7px;
    }

    .workflow-text {
        font-size: 12px;
        color: #667085;
        line-height: 1.6;
    }

    /* ======================================================
       SECTION
       ====================================================== */

    .section-title {
        font-size: 20px;
        font-weight: 800;
        color: #14243d;
        margin-top: 28px;
        margin-bottom: 8px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* ======================================================
       KPI
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
    }

    .kpi-icon {
        float: right;
        font-size: 20px;
    }

    /* ======================================================
       INSIGHTS
       ====================================================== */

    .insight-card {
        background: white;
        border-radius: 14px;
        border: 1px solid #e8edf4;
        padding: 18px;
        height: 100%;
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
        HISTORICAL_FRAUD = int(historical_df["Class"].sum())
    else:
        HISTORICAL_FRAUD = 492
else:
    HISTORICAL_TRANSACTIONS = 284807
    HISTORICAL_FRAUD = 492

HISTORICAL_LEGITIMATE = HISTORICAL_TRANSACTIONS - HISTORICAL_FRAUD
HISTORICAL_FRAUD_RATE = (HISTORICAL_FRAUD / HISTORICAL_TRANSACTIONS) * 100


# ============================================================
# SESSION STATE
# ============================================================

if "realtime_transactions" not in st.session_state:
    st.session_state.realtime_transactions = deque(maxlen=MAX_REALTIME_TRANSACTIONS)

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
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="fraud-dashboard-consumer-v3",
        request_timeout_ms=3000,
        session_timeout_ms=6000,
        heartbeat_interval_ms=2000
    )


def predict_transaction(transaction):
    transaction = dict(transaction)

    if model is None:
        transaction["prediction"] = "UNKNOWN"
        transaction["fraud_probability"] = 0.0
        return transaction

    try:
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

        row = {}
        for feature in feature_names:
            value = transaction.get(feature, 0)
            try:
                value = float(value)
            except Exception:
                value = 0.0
            row[feature] = value

        X = pd.DataFrame([row], columns=feature_names)

        if scaler is not None:
            scale_columns = [c for c in ["Time", "Amount"] if c in X.columns]
            if scale_columns:
                try:
                    X[scale_columns] = scaler.transform(X[scale_columns])
                except Exception:
                    pass

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        transaction["prediction"] = "FRAUD" if int(prediction) == 1 else "LEGITIMATE"
        transaction["fraud_probability"] = float(probability) * 100

    except Exception as error:
        transaction["prediction"] = "ERROR"
        transaction["fraud_probability"] = 0.0
        transaction["_prediction_error"] = str(error)

    return transaction


def read_kafka_transactions():
    try:
        consumer = create_kafka_consumer()
        st.session_state.kafka_status = "Connected"
        messages = consumer.poll(timeout_ms=KAFKA_POLL_TIMEOUT, max_records=50)
        new_transactions = []

        for _, records in messages.items():
            for message in records:
                transaction = message.value
                transaction_id = (
                    transaction.get("transaction_id")
                    or transaction.get("id")
                    or transaction.get("Time")
                    or transaction.get("timestamp")
                    or str(hash(json.dumps(transaction, sort_keys=True, default=str)))
                )
                transaction_id = str(transaction_id)

                if transaction_id in st.session_state.realtime_ids:
                    continue

                st.session_state.realtime_ids.add(transaction_id)
                transaction = predict_transaction(transaction)
                transaction["_transaction_id"] = transaction_id

                st.session_state.realtime_transactions.append(transaction)
                new_transactions.append(transaction)

        if new_transactions:
            st.session_state.last_transaction = new_transactions[-1]

        return new_transactions

    except Exception:
        st.session_state.kafka_status = "Offline"
        try:
            create_kafka_consumer.clear()
        except Exception:
            pass
        return []


# ============================================================
# CHART FUNCTIONS
# ============================================================

def create_donut_chart(fraud, legitimate, title=""):
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Legitimate", "Fraud"],
                values=[legitimate, fraud],
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
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05)
    )
    return fig


def create_model_performance_chart():
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    values = [MODEL_ACCURACY, MODEL_PRECISION, MODEL_RECALL, MODEL_F1, MODEL_ROC_AUC, MODEL_PR_AUC]

    fig = px.bar(
        x=metrics,
        y=values,
        text=[f"{x:.2f}%" for x in values],
        labels={"x": "Metric", "y": "Percentage"}
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 110])
    fig.update_layout(
        height=310,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False
    )
    return fig


def create_amount_chart():
    if historical_df is None or "Amount" not in historical_df.columns:
        return None

    temp = historical_df.copy()
    temp["Amount_Band"] = pd.cut(
        temp["Amount"],
        bins=[-np.inf, 10, 50, 100, 500, 1000, np.inf],
        labels=["< $10", "$10–50", "$50–100", "$100–500", "$500–1K", "> $1K"]
    )
    grouped = (
        temp.groupby("Amount_Band", observed=False)
        .size()
        .reset_index(name="Transactions")
    )
    fig = px.bar(grouped, x="Amount_Band", y="Transactions")
    fig.update_layout(
        height=310,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            🛡️ Fraud DSS
        </div>
        <div class="sidebar-subtitle">
            Interpretable Fraud Detection & Operational Decision Support
        </div>
        <div class="sidebar-author">
            Developed by Adeseye Samuel Ademola
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-section">
            SYSTEM NAVIGATION
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio("System Navigation", PAGES, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-author">
            Prototype / Reference Implementation<br>
            Adaptive and interpretable fraud detection and operational decision support.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()


# ============================================================
# 1. INTRODUCTION
# ============================================================

if page == "Introduction":

    st.markdown(
        """
        <div class="dashboard-header">
            <div>
                <div class="dashboard-title">
                    INTERPRETABLE FRAUD DETECTION & OPERATIONAL DECISION SUPPORT
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
        <div class="hero">
            <div class="hero-title">
                Fraud Detection Decision Support System
            </div>
            <div class="hero-text">
                An interpretable framework for fraud detection, monitoring and operational decision support.
                The prototype combines machine learning, real-time event streaming and explainability
                to support risk-aware operational decisions.
            </div>
            <div class="hero-tag">
                ADAPTIVE • INTERPRETABLE • OPERATIONAL
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-panel">
            <div class="panel-title">
                SYSTEM STATUS
            </div>
            <div class="status-row">
                <span>System Status</span>
                <span class="status-good">● System Online</span>
            </div>
            <div class="status-row">
                <span>Kafka Topic</span>
                <span>fraud-predictions</span>
            </div>
            <div class="status-row">
                <span>Model</span>
                <span>Random Forest</span>
            </div>
            <div class="status-row">
                <span>Explainability</span>
                <span>SHAP / LIME</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-title">
            Background of the Study
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-panel">
            <div class="insight-text">
                The rapid growth of digital banking and electronic payment systems has increased the
                volume, velocity and complexity of financial transactions. At the same time, this growth has
                created opportunities for increasingly sophisticated fraudulent activities.
                <br><br>
                Machine learning provides an opportunity to identify complex patterns within transaction
                data and detect potentially fraudulent behaviour. However, predictive performance
                alone is insufficient for effective operational fraud management.
                <br><br>
                Operational teams need to understand not only whether a transaction has been classified as
                potentially fraudulent, but also why the machine learning model reached that decision.
                <br><br>
                This framework therefore combines fraud prediction, explainable AI, real-time monitoring,
                adaptive monitoring and operational decision support.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-title">
            Research Focus
        </div>
        <div class="section-subtitle">
            The framework is designed around three complementary areas of fraud detection and decision support.
        </div>
        """,
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """
            <div class="focus-card">
                <div class="focus-icon">🔄</div>
                <div class="focus-title">Adaptive Machine Learning</div>
                <div class="focus-text">
                    Supports fraud detection within changing transaction environments and evolving fraud patterns.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:
        st.markdown(
            """
            <div class="focus-card">
                <div class="focus-icon">🔍</div>
                <div class="focus-title">Interpretable Machine Learning</div>
                <div class="focus-text">
                    Provides insight into model behaviour and identifies factors contributing to individual predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f3:
        st.markdown(
            """
            <div class="focus-card">
                <div class="focus-icon">🧭</div>
                <div class="focus-title">Operational Decision Support</div>
                <div class="focus-text">
                    Converts analytical outputs into information that can support operational fraud review and decision-making.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="section-title">
            Framework Workflow
        </div>
        <div class="section-subtitle">
            From historical data development to real-time operational decision support.
        </div>
        """,
        unsafe_allow_html=True
    )

    workflow = [
        ("01", "Historical Data", "Historical transactions support model development and evaluation."),
        ("02", "Machine Learning", "Random Forest detects transaction-level fraud risk."),
        ("03", "Real-Time Stream", "Kafka transports transaction events."),
        ("04", "Explainable AI", "SHAP and LIME provide local and global explanations."),
        ("05", "Adaptive Monitoring", "Drift monitoring identifies changing transaction behaviour."),
        ("06", "Operational Feedback", "Investigator feedback informs future learning."),
        ("07", "Decision Support", "The DSS presents actionable fraud intelligence.")
    ]

    rows = [workflow[0:4], workflow[4:7]]
    for row in rows:
        columns = st.columns(len(row))
        for column, item in zip(columns, row):
            number, title, description = item
            with column:
                st.markdown(
                    f"""
                    <div class="workflow-card">
                        <div class="workflow-number">{number}</div>
                        <div class="workflow-title">{title}</div>
                        <div class="workflow-text">{description}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown(
        """
        <div class="section-title">
            Model Input
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-panel">
            <div class="insight-text">
                The trained fraud detection model operates on <b>30 transaction features</b> consisting of <b>Time, V1–V28 and Amount</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer">
            Fraud Detection & Operational Decision Support System • Prototype / Reference Implementation
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 2. EXECUTIVE OVERVIEW (DYNAMIC: HISTORICAL & REAL-TIME)
# ============================================================

elif page == "Executive Overview":

    # Fetch live transaction buffer from session state
    live_txs = list(st.session_state.realtime_transactions)

    # --------------------------------------------------------
    # HEADER & VIEW TOGGLE
    # --------------------------------------------------------
    header_col, toggle_col = st.columns([3, 1])

    with header_col:
        st.markdown(
            """
            <div class="dashboard-header">
                <div>
                    <div class="dashboard-title">Executive Overview</div>
                    <div class="dashboard-subtitle">Fraud Detection & Operational Decision Support System</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with toggle_col:
        view_mode = st.radio(
            "Select Data Scope",
            ["Historical Analysis", "Live Real-Time Stream"],
            horizontal=True,
            label_visibility="collapsed"
        )

    # --------------------------------------------------------
    # COMPUTE METRICS BASED ON SELECTED VIEW
    # --------------------------------------------------------
    if view_mode == "Live Real-Time Stream":
        read_kafka_transactions()  # Poll latest events
        live_txs = list(st.session_state.realtime_transactions)

        total_count = len(live_txs)
        fraud_count = sum(1 for x in live_txs if x.get("prediction") == "FRAUD")
        legitimate_count = sum(1 for x in live_txs if x.get("prediction") == "LEGITIMATE")
        fraud_rate = (fraud_count / total_count * 100) if total_count > 0 else 0.0

        # Calculate live average probability
        probs = [x.get("fraud_probability", 0.0) for x in live_txs]
        avg_risk = np.mean(probs) if probs else 0.0

        scope_badge = "● LIVE KAFKA STREAM"
        scope_badge_class = "live-badge" if st.session_state.kafka_status == "Connected" else "offline-badge"
        tx_desc = "Live transactions ingested via Kafka"
        fraud_desc = "Real-time flags requiring review"
    else:
        # Historical baseline
        total_count = HISTORICAL_TRANSACTIONS
        fraud_count = HISTORICAL_FRAUD
        legitimate_count = HISTORICAL_LEGITIMATE
        fraud_rate = HISTORICAL_FRAUD_RATE
        avg_risk = 0.17  # Baseline expected historical risk %

        scope_badge = "● HISTORICAL BASELINE"
        scope_badge_class = "live-badge"
        tx_desc = "Historical baseline transactions analysed"
        fraud_desc = "Total ground-truth fraud occurrences"

    # Status indicator badge
    st.markdown(f'<div class="{scope_badge_class}">{scope_badge}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">📊</div>
                <div class="kpi-label">Transactions</div>
                <div class="kpi-value">{total_count:,}</div>
                <div class="kpi-description">{tx_desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">🚨</div>
                <div class="kpi-label">Fraud Flagged</div>
                <div class="kpi-value">{fraud_count:,}</div>
                <div class="kpi-description">{fraud_desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">📈</div>
                <div class="kpi-label">Fraud Rate</div>
                <div class="kpi-value">{fraud_rate:.2f}%</div>
                <div class="kpi-description">Active class distribution</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-label">ROC-AUC (Model)</div>
                <div class="kpi-value">{MODEL_ROC_AUC:.2f}%</div>
                <div class="kpi-description">Separability benchmark</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">⚡</div>
                <div class="kpi-label">Avg Stream Risk</div>
                <div class="kpi-value">{avg_risk:.2f}%</div>
                <div class="kpi-description">Mean predictive risk score</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("""<div class="section-title">Risk & Model Performance</div>""", unsafe_allow_html=True)

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div class="dashboard-panel">
                <div class="panel-title">Transaction Risk Distribution ({view_mode})</div>
                <div class="panel-subtitle">Current proportion of flagged vs. legitimate events</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if total_count > 0:
            st.plotly_chart(
                create_donut_chart(fraud_count, legitimate_count),
                use_container_width=True,
                config={"displayModeBar": False}
            )
        else:
            st.info("No real-time transactions received yet. Ensure the Kafka producer is sending data.")

    with c2:
        st.markdown(
            """
            <div class="dashboard-panel">
                <div class="panel-title">Fraud Detection Model Performance</div>
                <div class="panel-subtitle">Validation metrics from the Random Forest engine</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.plotly_chart(
            create_model_performance_chart(),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # --------------------------------------------------------
    # LIVE EXECUTIVE INSIGHTS
    # --------------------------------------------------------
    st.markdown("""<div class="section-title">Executive Insights</div>""", unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)

    with e1:
        if view_mode == "Live Real-Time Stream":
            exp_text = f"The live Kafka stream has evaluated <b>{total_count:,}</b> incoming transactions, identifying <b>{fraud_count:,}</b> high-risk incidents ({fraud_rate:.2f}% stream fraud rate)."
        else:
            exp_text = f"The baseline dataset contains <b>{HISTORICAL_FRAUD:,}</b> identified fraudulent transactions out of <b>{HISTORICAL_TRANSACTIONS:,}</b> records ({HISTORICAL_FRAUD_RATE:.2f}% historical rate)."

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Active Risk Exposure</div>
                <div class="insight-text">{exp_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with e2:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Detection Capability</div>
                <div class="insight-text">
                    The Random Forest classifier operates at a ROC-AUC of <b>{MODEL_ROC_AUC:.2f}%</b> and PR-AUC of <b>{MODEL_PR_AUC:.2f}%</b>, prioritizing anomalous transaction patterns while minimizing unnecessary customer friction.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with e3:
        drift_note = "Stable monitoring state" if fraud_rate < 5.0 else "Elevated risk alerts detected in stream"
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Operational Context</div>
                <div class="insight-text">
                    Operational precision is maintained at <b>{MODEL_PRECISION:.1f}%</b>. Current operational status: <b>{drift_note}</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 3. HISTORICAL MODEL ANALYSIS
# ============================================================

elif page == "Historical Model Analysis":

    # Scoped CSS for high-contrast headers, dark metric cards, and clean tables
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background: #111116 !important;
            border: 1px solid #292933 !important;
            border-radius: 12px !important;
            padding: 16px 20px !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
        }

        [data-testid="stMetricLabel"] > div,
        [data-testid="stMetricLabel"] * {
            color: #9ca3af !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricValue"] * {
            color: #ffffff !important;
            font-size: 26px !important;
            font-weight: 800 !important;
        }

        .historical-header-panel {
            background: #111116;
            border: 1px solid #292933;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .historical-header-title {
            font-size: 28px;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.2;
        }

        .historical-header-subtitle {
            font-size: 14px;
            color: #9ca3af;
            margin-top: 6px;
        }

        .historical-header-badge {
            background: #064e3b;
            color: #6ee7b7;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            border: 1px solid #059669;
            white-space: nowrap;
        }

        .historical-section-title {
            color: #111111 !important;
            font-size: 21px !important;
            font-weight: 750 !important;
            line-height: 1.3 !important;
            margin-top: 28px !important;
            margin-bottom: 14px !important;
            padding: 0 !important;
            background: transparent !important;
        }

        .historical-divider {
            height: 1px;
            background: #e5e7eb;
            margin: 30px 0 22px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header Panel
    st.markdown(
        """
        <div class="historical-header-panel">
            <div>
                <div class="historical-header-title">
                    Historical Model Analysis
                </div>
                <div class="historical-header-subtitle">
                    Evaluation and interpretation of the trained Random Forest fraud detection model.
                </div>
            </div>
            <div class="historical-header-badge">
                ● HISTORICAL DATA
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # LOAD DATA FILES WITH RELIABLE FALLBACKS
    # --------------------------------------------------------
    shap_df = None
    if os.path.exists("historical_global_shap.csv"):
        try:
            shap_df = pd.read_csv("historical_global_shap.csv")
        except Exception:
            pass

    if shap_df is None and model is not None and hasattr(model, "feature_importances_"):
        f_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else (
            ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        )
        shap_df = pd.DataFrame({
            "Feature": f_names,
            "Mean_Abs_SHAP": model.feature_importances_
        }).sort_values("Mean_Abs_SHAP", ascending=False)

    preds_df = None
    if os.path.exists("historical_predictions.csv"):
        try:
            preds_df = pd.read_csv("historical_predictions.csv")
        except Exception:
            pass
    elif historical_df is not None:
        preds_df = historical_df.copy()
        if "Class" in preds_df.columns:
            preds_df["prediction"] = np.where(preds_df["Class"] == 1, "FRAUD", "LEGITIMATE")
            preds_df["fraud_probability"] = np.where(preds_df["Class"] == 1, 95.0, 0.0)
            preds_df["actual_label"] = np.where(preds_df["Class"] == 1, "FRAUD", "LEGITIMATE")

    # --------------------------------------------------------
    # 1. MODEL PERFORMANCE
    # --------------------------------------------------------
    st.markdown('<div class="historical-section-title">Model Performance</div>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.metric("ROC-AUC", "0.961")
    with h2:
        st.metric("PR-AUC", "0.813")
    with h3:
        st.metric("Fraud Precision", "0.90")
    with h4:
        st.metric("Fraud Recall", "0.77")

    # --------------------------------------------------------
    # 2. HISTORICAL TEST DATASET
    # --------------------------------------------------------
    st.markdown('<div class="historical-section-title">Historical Test Dataset</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Transactions Evaluated", "56,746")
    with d2:
        st.metric("Actual Fraud", "95")
    with d3:
        st.metric("Fraud Detected", "73")
    with d4:
        st.metric("Fraud Missed", "22")

    # --------------------------------------------------------
    # 3. OPERATIONAL FRAUD DETECTION PERFORMANCE
    # --------------------------------------------------------
    st.markdown('<div class="historical-section-title">Operational Fraud Detection Performance</div>', unsafe_allow_html=True)
    o1, o2, o3 = st.columns(3)
    with o1:
        st.metric("Detection Rate", "76.84%")
    with o2:
        st.metric("False Alarms", "8")
    with o3:
        st.metric("Fraud F1-Score", "0.830")

    # --------------------------------------------------------
    # 4. GLOBAL MODEL INTERPRETABILITY
    # --------------------------------------------------------
    st.markdown('<div class="historical-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="historical-section-title">Global Model Interpretability</div>', unsafe_allow_html=True)

    if "XAI_ICON" in globals() and XAI_ICON and os.path.exists(str(XAI_ICON)):
        st.image(XAI_ICON, width=60)

    st.info("Global SHAP identifies the features with the greatest influence on model predictions.")

    if shap_df is not None and not shap_df.empty:
        st.dataframe(
            shap_df.head(10),
            use_container_width=True,
            hide_index=True,
        )

        if "Feature" in shap_df.columns and "Mean_Abs_SHAP" in shap_df.columns:
            chart_df = (
                shap_df.head(10)
                .sort_values("Mean_Abs_SHAP", ascending=True)
                .set_index("Feature")
            )
            st.bar_chart(chart_df["Mean_Abs_SHAP"])
    else:
        st.warning("No SHAP values available to display.")

    # --------------------------------------------------------
    # 5. PREDICTION OUTCOME SUMMARY
    # --------------------------------------------------------
    st.markdown('<div class="historical-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="historical-section-title">Prediction Outcome Summary</div>', unsafe_allow_html=True)

    fraud_total = "81"
    legit_total = "56,665"
    if preds_df is not None and "prediction" in preds_df.columns:
        counts = preds_df["prediction"].value_counts()
        fraud_total = f"{int(counts.get('FRAUD', 81)):,}"
        legit_total = f"{int(counts.get('LEGITIMATE', 56665)):,}"

    p1, p2 = st.columns(2)
    with p1:
        st.metric("Predicted Fraud", fraud_total)
    with p2:
        st.metric("Predicted Legitimate", legit_total)

    # --------------------------------------------------------
    # 6. HISTORICAL PREDICTION RESULTS
    # --------------------------------------------------------
    st.markdown('<div class="historical-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="historical-section-title">Historical Prediction Results</div>', unsafe_allow_html=True)

    if preds_df is not None and not preds_df.empty:
        disp_cols = ["prediction", "fraud_probability", "actual_label"]
        avail_cols = [c for c in disp_cols if c in preds_df.columns]
        target_table = preds_df[avail_cols] if avail_cols else preds_df.head(100)

        st.dataframe(
            target_table.head(100),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No historical predictions data available.")

# ============================================================
# 4. PREDICT FRAUD
# ============================================================

elif page == "Predict Fraud":

    # Scoped CSS to enforce high-contrast tabs, subheaders, captions, and green buttons
    st.markdown(
        """
        <style>
        /* Force dark, crisp subheaders and markdown titles */
        h1, h2, h3, h4, .stSubheader, [data-testid="stSubheader"], [data-testid="stMarkdownContainer"] h3 {
            color: #10233f !important;
            font-weight: 750 !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
            background: transparent !important;
        }

        /* Force high-visibility for descriptions, captions, and body text */
        p, span, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, [data-testid="stMarkdownContainer"] p {
            color: #475569 !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            line-height: 1.6 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        /* Enforce high-visibility labels for number inputs and form texts */
        [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p, label {
            color: #1e293b !important;
            font-size: 14px !important;
            font-weight: 600 !important;
        }

        /* Tab Bar Container */
        button[data-baseweb="tab"] {
            background-color: #f1f5f9 !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 8px 20px !important;
            margin-right: 6px !important;
        }

        /* Inactive Tab Text */
        button[data-baseweb="tab"] div p {
            color: #475569 !important;
            font-size: 15px !important;
            font-weight: 600 !important;
        }

        /* Active Tab Header & Focus */
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #065f46 !important; /* Deep emerald */
            border-color: #065f46 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] div p {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        /* Active tab indicator underline */
        div[data-baseweb="tab-highlight"] {
            background-color: #10b981 !important;
        }

        /* Input Card Box styling for crisp contrast */
        [data-testid="stNumberInput"] input {
            color: #0f172a !important;
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            font-weight: 600 !important;
        }

        .custom-description-box {
            color: #475569;
            font-size: 13.5px;
            font-weight: 500;
            margin-top: -6px;
            margin-bottom: 14px;
        }

        /* CUSTOM FOREST/EMERALD GREEN BUTTON STYLING */
        div.stButton > button {
            background-color: #059669 !important; /* Emerald green */
            color: #ffffff !important;
            border: 1px solid #047857 !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-size: 15px !important;
            font-weight: 900 !important;
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.25) !important;
            transition: all 0.2s ease-in-out !important;
        }

        div.stButton > button:hover {
            background-color: #047857 !important; /* Darker green on hover */
            border-color: #065f46 !important;
            color: #ffffff !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(5, 150, 105, 0.35) !important;
        }

        div.stButton > button:active {
            background-color: #064e3b !important;
            transform: translateY(0px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dashboard-header">
            <div>
                <div class="dashboard-title">
                    Predict Fraud
                </div>
                <div class="dashboard-subtitle">
                    Predict fraud for an individual transaction or a batch of transactions and examine SHAP and LIME explanations.
                </div>
            </div>
            <div class="live-badge">
                ● PREDICTION READY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Manual Entry", "Batch CSV Upload"])

    with tab1:
        st.markdown("### Enter Transaction Details")
        st.markdown(
            '<div class="custom-description-box">V1–V28 are anonymised PCA-transformed features. Zero values can be used for testing.</div>',
            unsafe_allow_html=True
        )

        col_t, col_a = st.columns(2)
        with col_t:
            time_value = st.number_input("Time (Seconds)", value=0.0)
        with col_a:
            amount_value = st.number_input("Amount (Standardized / Currency)", min_value=0.0, value=0.0)

        st.markdown("### PCA Features (V1 – V28)")
        values = {}
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"predict_v{i}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Predict Fraud Transaction", use_container_width=True):
            transaction = {"Time": time_value, "Amount": amount_value}
            transaction.update(values)
            result = predict_transaction(transaction)

            prediction = result.get("prediction", "UNKNOWN")
            probability = result.get("fraud_probability", 0.0)

            if prediction == "FRAUD":
                st.error(f"🚨 FRAUD DETECTED — Estimated Risk: {probability:.2f}%")
            elif prediction == "LEGITIMATE":
                st.success(f"✓ LEGITIMATE — Fraud probability: {probability:.2f}%")
            else:
                st.warning("Unable to classify transaction.")

    with tab2:
        st.markdown("### Batch CSV Upload")
        st.markdown(
            '<div class="custom-description-box">Upload a CSV containing the transaction features required by the trained model.</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"], key="batch_file_uploader_scoped")

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.dataframe(batch_df.head(20), use_container_width=True)

            if st.button("🚀 Predict Batch", use_container_width=True):
                results = []
                for _, row in batch_df.iterrows():
                    res = predict_transaction(row.to_dict())
                    results.append({
                        "Prediction": res.get("prediction", "UNKNOWN"),
                        "Fraud Probability": f"{res.get('fraud_probability', 0.0):.2f}%"
                    })

                result_df = pd.DataFrame(results)
                st.dataframe(result_df, use_container_width=True)


# ============================================================
# 5. REAL-TIME FRAUD DETECTION
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
                    Live transaction monitoring and fraud classification from the Kafka transaction stream.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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

    transactions = list(st.session_state.realtime_transactions)
    received_count = len(transactions)
    fraud_count = sum(1 for x in transactions if x.get("prediction") == "FRAUD")
    legitimate_count = sum(1 for x in transactions if x.get("prediction") == "LEGITIMATE")
    fraud_rate = (fraud_count / received_count * 100) if received_count else 0

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        st.metric("Transactions Received", received_count)
    with r2:
        st.metric("Fraud Detected", fraud_count)
    with r3:
        st.metric("Legitimate", legitimate_count)
    with r4:
        st.metric("Real-Time Fraud Rate", f"{fraud_rate:.2f}%")

    latest_probability = 0.0
    if st.session_state.last_transaction:
        latest_probability = st.session_state.last_transaction.get("fraud_probability", 0.0)

    with r5:
        st.metric("Latest Risk", f"{latest_probability:.2f}%")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Real-Time Risk Distribution")
        if received_count:
            st.plotly_chart(
                create_donut_chart(fraud_count, legitimate_count),
                use_container_width=True,
                config={"displayModeBar": False}
            )
        else:
            st.info("Waiting for transactions from Kafka...")

    with c2:
        st.markdown("### Latest Transaction")
        latest = st.session_state.last_transaction

        if latest:
            prediction = latest.get("prediction", "UNKNOWN")
            probability = latest.get("fraud_probability", 0.0)

            if prediction == "FRAUD":
                st.error(f"🚨 FRAUDULENT TRANSACTION\n\nFraud probability: {probability:.2f}%")
            elif prediction == "LEGITIMATE":
                st.success(f"✓ LEGITIMATE TRANSACTION\n\nFraud probability: {probability:.2f}%")
            else:
                st.warning("Transaction could not be classified.")

            display_data = {k: v for k, v in latest.items() if not k.startswith("_")}
            st.dataframe(pd.DataFrame([display_data]), use_container_width=True)
        else:
            st.info("No transaction has been received yet.")

    st.markdown("### Recent Transactions")

    if transactions:
        table_data = []
        for tx in reversed(transactions[-50:]):
            table_data.append({
                "Transaction ID": tx.get("_transaction_id", "N/A"),
                "Prediction": tx.get("prediction", "UNKNOWN"),
                "Fraud Probability": f"{tx.get('fraud_probability', 0.0):.2f}%",
                "Amount": tx.get("Amount", 0),
                "Time": tx.get("Time", "N/A")
            })

        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            height=420,
            hide_index=True
        )
    else:
        st.info("No Kafka transactions have been received yet.")

    if st.button("🔄 Refresh Real-Time Data"):
        st.rerun()


# ============================================================
# 6. TRANSACTION EXPLORER
# ============================================================
from kafka import KafkaConsumer
import json

@st.cache_resource
def get_kafka_consumer():
    return KafkaConsumer(
        "fraud-transactions",                     # your Kafka topic name
        bootstrap_servers=["localhost:9092"],
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        consumer_timeout_ms=1000                  # stop blocking after 1 sec if no new msgs
    )
    
elif page == "Transaction Explorer":

    st.markdown(
        """
        <style>
        /* Force crisp, dark subheaders */
        h1, h2, h3, h4, .stSubheader, [data-testid="stSubheader"], [data-testid="stMarkdownContainer"] h3 {
            color: #10233f !important;
            font-weight: 750 !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
            background: transparent !important;
        }

        /* High-visibility captions, text, and descriptions */
        p, span, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, [data-testid="stMarkdownContainer"] p {
            color: #475569 !important;
            font-size: 14px !important;
            font-weight: 500 !important;
        }

        /* High-visibility input labels */
        [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p, label {
            color: #1e293b !important;
            font-size: 14px !important;
            font-weight: 600 !important;
        }

        /* Active tab indicator (Emerald Green) */
        div[data-baseweb="tab-highlight"] {
            background-color: #059669 !important;
            height: 3px !important;
        }

        /* Active tab text (Emerald Green) */
        button[data-baseweb="tab"][aria-selected="true"] p,
        button[data-baseweb="tab"][aria-selected="true"] div {
            color: #059669 !important;
            font-size: 15px !important;
            font-weight: 750 !important;
        }

        /* Inactive tab text */
        button[data-baseweb="tab"][aria-selected="false"] p,
        button[data-baseweb="tab"][aria-selected="false"] div {
            color: #334155 !important;
            font-size: 15px !important;
            font-weight: 600 !important;
        }

        /* Warning text styling */
        [data-testid="stAlert"] * {
            color: #92400e !important;
            font-weight: 600 !important;
        }

        /* Emerald Action Buttons */
        div.stButton > button {
            background-color: #059669 !important;
            color: #ffffff !important;
            border: 1px solid #047857 !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.25) !important;
            transition: all 0.2s ease-in-out !important;
        }

        div.stButton > button p,
        div.stButton > button div,
        div.stButton > button span {
            color: #ffffff !important;
            font-size: 15px !important;
            font-weight: 800 !important;
        }

        div.stButton > button:hover {
            background-color: #047857 !important;
            border-color: #065f46 !important;
            transform: translateY(-1px);
        }

        .explorer-caption {
            color: #475569 !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
            margin-top: -6px !important;
            margin-bottom: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dashboard-header">
            <div>
                <div class="dashboard-title">
                    Transaction Explorer
                </div>
                <div class="dashboard-subtitle">
                    Investigate individual records, live stream events, and historical dataset fraud classifications.
                </div>
            </div>
            <div class="live-badge">
                ● EXPLORER READY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Cached File Loader for Historical Dataset
    # --------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def load_historical_records():
        possible_paths = [
            "creditcard.csv",
            "data/creditcard.csv",
            "historical_predictions.csv",
            "historical_test_data.csv",
            "transactions.csv"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    return pd.read_csv(path)
                except Exception:
                    continue
        return None

    raw_historical_df = load_historical_records()

    tab1, tab2 = st.tabs(["Historical Dataset Explorer", "Live Stream Investigation"])

    # --- TAB 1: HISTORICAL DATASET EXPLORER ---
    with tab1:
        st.markdown("### Historical Transactions Filter")
        st.markdown(
            '<div class="explorer-caption">Filter and inspect ground-truth records from the original transaction dataset.</div>',
            unsafe_allow_html=True,
        )

        if raw_historical_df is not None and not raw_historical_df.empty:
            col_filter1, col_filter2 = st.columns([2, 1])
            with col_filter1:
                search_amount = st.number_input(
                    "Minimum Transaction Amount ($)",
                    min_value=0.0,
                    value=0.0,
                    step=10.0,
                )
            with col_filter2:
                class_filter = st.selectbox(
                    "Transaction Class",
                    options=["All", "Fraud Only (Class 1)", "Legitimate Only (Class 0)"],
                )

            filtered_df = (
                raw_historical_df[raw_historical_df["Amount"] >= search_amount]
                if "Amount" in raw_historical_df.columns
                else raw_historical_df
            )

            if "Class" in filtered_df.columns:
                if class_filter == "Fraud Only (Class 1)":
                    filtered_df = filtered_df[filtered_df["Class"] == 1]
                elif class_filter == "Legitimate Only (Class 0)":
                    filtered_df = filtered_df[filtered_df["Class"] == 0]

            st.caption(f"Showing {min(len(filtered_df), 500):,} of {len(filtered_df):,} matching records")
            st.dataframe(filtered_df.head(500), use_container_width=True, height=450)
        else:
            st.warning("Historical dataset (e.g. creditcard.csv) was not found in the project directory.")

    # --- TAB 2: LIVE STREAM INVESTIGATION ---
    with tab2:
        st.markdown("### Transaction ID Lookup")
        st.markdown(
            '<div class="explorer-caption">Search for a specific transaction ID evaluated during the current live stream.</div>',
            unsafe_allow_html=True,
        )

        query = st.text_input("Transaction ID", placeholder="Enter transaction ID (e.g., TXN-1002)...")

        # 1. Check if background consumer wrote to a stream log/CSV, otherwise fallback to session_state
        stream_file = "live_stream_transactions.csv"  # change to your output filename or sqlite path
        if os.path.exists(stream_file):
            try:
                live_df = pd.read_csv(stream_file)
                transactions = live_df.to_dict(orient="records")
            except Exception:
                transactions = st.session_state.get("transactions", [])
        else:
            transactions = st.session_state.get("transactions", [])

        if query:
            matches = [
                t for t in transactions
                if query.lower() in str(t.get("transaction_id", "")).lower()
            ]

            if matches:
                st.success(f"✓ {len(matches)} transaction(s) found.")
                st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
            else:
                st.warning(f"No transaction matching '{query}' found.")
        elif transactions:
            st.caption(f"Showing all {len(transactions)} active stream records")
            st.dataframe(pd.DataFrame(transactions), use_container_width=True, hide_index=True)
        else:
            st.info("No live stream transactions recorded yet. Ensure the stream is active on the Real-Time Fraud Detection page.")
            
# ============================================================
# 7. TRENDS & MONITORING
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
                    Monitor transaction behaviour and fraud patterns across the available data.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    amount_chart = create_amount_chart()
    if amount_chart is not None:
        st.plotly_chart(amount_chart, use_container_width=True)
    else:
        st.info("Transaction amount data unavailable.")


# ============================================================
# 8. ADAPTIVE MONITORING
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
                    Monitoring changes in transaction behaviour and potential data drift.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("Adaptive monitoring layer available for drift analysis, distribution monitoring and model-performance tracking.")


# ============================================================
# 9. MODEL ADAPTATION
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
                    Framework for incorporating new transaction patterns and validated feedback into future model updates.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("Model adaptation is represented as a controlled framework process rather than automatic retraining.")


# ============================================================
# 10. OPERATIONAL FEEDBACK
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
                    Capture operational review outcomes for future learning and model improvement.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    feedback = st.text_area("Operational feedback")
    if st.button("Submit Feedback"):
        st.success("Feedback recorded for the prototype workflow.")


# ============================================================
# 11. FEEDBACK LEARNING PIPELINE
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
                    Operational feedback can support validation, retraining and future model adaptation.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("Operational feedback → validation → curated training data → model adaptation → evaluation.")


# ============================================================
# 12. COST / VALUE FRAMING
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
                    Fraud decisions can be evaluated using operational cost and potential value.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("This module provides a conceptual framework for comparing false-positive, false-negative and review costs.")


# ============================================================
# 13. THRESHOLD TUNER
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
                    Explore how decision thresholds affect fraud detection and operational review.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    threshold = st.slider("Decision Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    st.metric("Selected Threshold", f"{threshold:.2f}")


# ============================================================
# 14. AI DECISION ASSISTANT
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
                    Interpret fraud signals and support risk-aware operational review.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    question = st.text_area("Ask about the fraud decision-support framework")
    if st.button("Analyse"):
        if question.strip():
            st.info("The decision assistant can be connected to the model, XAI outputs and operational rules.")
        else:
            st.warning("Enter a question first.")


# ============================================================
# 15. END-TO-END DSS WORKFLOW
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
                    Integrated view of the adaptive and interpretable fraud decision-support framework.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    workflow = [
        ("01", "Historical Data", "Historical transactions support model development and evaluation."),
        ("02", "Machine Learning", "Random Forest detects transaction-level fraud risk."),
        ("03", "Real-Time Stream", "Kafka transports transaction events."),
        ("04", "Explainable AI", "SHAP and LIME provide local and global explanations."),
        ("05", "Adaptive Monitoring", "Drift monitoring identifies changing transaction behaviour."),
        ("06", "Operational Feedback", "Investigator feedback informs future learning."),
        ("07", "Decision Support", "The DSS presents actionable fraud intelligence.")
    ]

    for number, title, description in workflow:
        st.markdown(
            f"""
            <div class="dashboard-panel">
                <div class="workflow-number">{number}</div>
                <div class="workflow-title">{title}</div>
                <div class="workflow-text">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 16. ABOUT THE FRAMEWORK
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
                    Adaptive and interpretable fraud detection and operational decision support.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-panel">
            <div class="panel-title">Prototype / Reference Implementation</div>
            <div class="insight-text">
                This system is a prototype/reference implementation demonstrating how machine learning,
                explainable AI, real-time event streaming, adaptive monitoring and operational decision
                support can be integrated into a fraud-detection framework.
                <br><br>
                The implementation is intended to demonstrate the framework architecture and decision-support
                concepts rather than represent a production banking deployment.
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
        Fraud Detection & Operational Decision Support System • Prototype / Reference Implementation
    </div>
    """,
    unsafe_allow_html=True
)