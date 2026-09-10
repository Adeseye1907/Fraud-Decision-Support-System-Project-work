# ============================================================
# FRAUD DSS
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
# CONFIGURATION
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
# MODEL PERFORMANCE
# ============================================================

MODEL_ACCURACY = 100.00
MODEL_PRECISION = 90.00
MODEL_RECALL = 77.00
MODEL_F1 = 83.00
MODEL_ROC_AUC = 96.06
MODEL_PR_AUC = 81.27


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
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .sidebar-brand {
        font-size: 24px;
        font-weight: 800;
        color: white;
        margin-bottom: 6px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        line-height: 1.5;
        color: #b8c8dc !important;
        margin-bottom: 4px;
    }

    .sidebar-author {
        font-size: 11px;
        color: #91a7c2 !important;
        margin-bottom: 25px;
    }

    .sidebar-section {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.2px;
        color: #7893b2 !important;
        margin-top: 18px;
        margin-bottom: 8px;
    }

    section[data-testid="stSidebar"] .stRadio label {
        font-size: 13px;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
    }

    .sidebar-footer {
        margin-top: 30px;
        padding-top: 15px;
        border-top: 1px solid rgba(255,255,255,0.10);
        color: #8298b2 !important;
        font-size: 10px;
        line-height: 1.5;
    }

    /* ======================================================
       HEADER
       ====================================================== */

    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 20px;
    }

    .dashboard-title {
        font-size: 30px;
        font-weight: 800;
        color: #10233f;
        margin-bottom: 3px;
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

    .info-badge {
        background: #eff6ff;
        color: #1d4ed8;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid #bfdbfe;
        white-space: nowrap;
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
        font-size: 11px;
        color: #6b7280;
        font-weight: 700;
        margin-bottom: 9px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
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
       SECTIONS
       ====================================================== */

    .section-title {
        font-size: 18px;
        font-weight: 800;
        color: #14243d;
        margin-top: 25px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 15px;
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
        font-weight: 800;
        color: #172b4d;
        margin-bottom: 5px;
    }

    .panel-subtitle {
        font-size: 11px;
        color: #8a94a6;
        margin-bottom: 12px;
    }

    /* ======================================================
       STATUS ROW
       ====================================================== */

    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
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

    .status-neutral {
        color: #64748b;
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
       REAL-TIME ALERTS
       ====================================================== */

    .transaction-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e8edf4;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
    }

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
       WORKFLOW
       ====================================================== */

    .workflow-step {
        background: white;
        border: 1px solid #e5eaf1;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .workflow-number {
        display: inline-block;
        background: #10233f;
        color: white;
        width: 27px;
        height: 27px;
        border-radius: 50%;
        text-align: center;
        line-height: 27px;
        font-size: 12px;
        font-weight: 800;
        margin-right: 9px;
    }

    .workflow-title {
        font-weight: 800;
        color: #172b4d;
        font-size: 14px;
    }

    .workflow-description {
        color: #667085;
        font-size: 12px;
        line-height: 1.5;
        margin-top: 7px;
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
       SMALL INFO BOX
       ====================================================== */

    .research-note {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #64748b;
        border-radius: 10px;
        padding: 14px 16px;
        color: #475569;
        font-size: 12px;
        line-height: 1.6;
        margin: 12px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
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
# HISTORICAL DATA
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


if "threshold_value" not in st.session_state:

    st.session_state.threshold_value = 0.50


# ============================================================
# MODEL FEATURES
# ============================================================

DEFAULT_FEATURES = (
    ["Time", "Amount"]
    + [f"V{i}" for i in range(1, 29)]
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

        probability = model.predict_proba(X)[0][1]

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
# KAFKA CONSUMER
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

        consumer = create_kafka_consumer()

        consumer.close()

    except Exception:

        pass

    try:

        create_kafka_consumer.clear()

    except Exception:

        pass


# ============================================================
# KAFKA TRANSACTION READER
# ============================================================

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

                transaction = dict(
                    message.value
                )

                # ------------------------------------------------
                # More robust transaction identifier
                # ------------------------------------------------

                transaction_id = (
                    transaction.get("transaction_id")
                    or transaction.get("id")
                    or transaction.get("event_id")
                    or transaction.get("timestamp")
                )

                if transaction_id is None:

                    transaction_id = (
                        f"{message.partition}-"
                        f"{message.offset}"
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

        # --------------------------------------------------------
        # Keep duplicate ID memory bounded
        # --------------------------------------------------------

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

        try:

            create_kafka_consumer.clear()

        except Exception:

            pass

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


def create_historical_amount_chart():

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
# PAGE HEADER
# ============================================================

def page_header(
    title,
    subtitle,
    badge_text=None,
    badge_type="info"
):

    if badge_type == "live":

        badge_class = "live-badge"

    elif badge_type == "offline":

        badge_class = "offline-badge"

    else:

        badge_class = "info-badge"

    badge_html = ""

    if badge_text:

        badge_html = f"""
        <div class="{badge_class}">
            {badge_text}
        </div>
        """

    st.markdown(
        f"""
        <div class="dashboard-header">

            <div>

                <div class="dashboard-title">
                    {title}
                </div>

                <div class="dashboard-subtitle">
                    {subtitle}
                </div>

            </div>

            {badge_html}

        </div>
        """,
        unsafe_allow_html=True
    )


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

        <div class="sidebar-section">
            SYSTEM NAVIGATION
        </div>
        """,
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
        <div class="sidebar-footer">
            Prototype / Reference Implementation<br>
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

    page_header(
        "Fraud Detection Decision Support System",
        "An interpretable framework for fraud detection, monitoring and operational decision support.",
        "● FRAMEWORK",
        "info"
    )

    st.markdown(
        """
        <div class="dashboard-panel">

            <div class="panel-title">
                Welcome to the Fraud DSS
            </div>

            <div class="insight-text">

                This prototype demonstrates an adaptive and
                interpretable machine learning framework for
                detecting conventional and emerging fraud
                patterns while supporting operational
                decision-making.

                <br><br>

                The system combines historical machine learning,
                real-time event streaming, explainability,
                adaptive monitoring and operational feedback
                into a single decision-support environment.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Framework Components</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-title">
                    🧠 Intelligent Detection
                </div>

                <div class="insight-text">
                    Machine learning is used to classify
                    transactions according to their estimated
                    fraud risk.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with b:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-title">
                    🔎 Interpretable Decisions
                </div>

                <div class="insight-text">
                    Model behaviour and transaction-level risk
                    can be interpreted to support more transparent
                    operational decisions.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-title">
                    🔄 Adaptive Monitoring
                </div>

                <div class="insight-text">
                    Monitoring and feedback mechanisms provide a
                    pathway for identifying changing transaction
                    behaviour and supporting future adaptation.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Project Positioning</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="research-note">

        This implementation is a prototype / reference
        implementation of the proposed decision-support
        framework. It does not represent a production banking
        deployment. In an operational banking environment,
        synthetic transaction generation would be replaced by
        secure enterprise transaction sources and controlled
        event-streaming infrastructure.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

elif page == "Executive Overview":

    page_header(
        "Executive Dashboard",
        "Fraud Detection & Operational Decision Support System",
        "● SYSTEM READY",
        "live"
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
                    Fraud proportion in historical dataset
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

    st.markdown(
        '<div class="section-subtitle">Historical fraud exposure and evaluated model capability.</div>',
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
                    Historical fraud and legitimate transaction profile
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
                    Evaluation metrics from the Random Forest model
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
    # MODEL + PIPELINE
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
            '<div class="section-title">Decision Support Status</div>',
            unsafe_allow_html=True
        )

        kafka_active = (
            st.session_state.kafka_status
            == "Connected"
        )

        st.markdown(
            f"""
            <div class="dashboard-panel">

                <div class="status-row">
                    <span>Historical Data Layer</span>
                    <span class="status-good">
                        ● AVAILABLE
                    </span>
                </div>

                <div class="status-row">
                    <span>Random Forest Model</span>
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
                    <span>Real-Time Classification</span>
                    <span class="status-good">
                        ● AVAILABLE
                    </span>
                </div>

                <div class="status-row">
                    <span>Explainability Layer</span>
                    <span class="status-good">
                        ● AVAILABLE
                    </span>
                </div>

                <div class="status-row">
                    <span>Adaptive Monitoring</span>
                    <span class="status-good">
                        ● FRAMEWORK READY
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
                    identified fraudulent transactions out of
                    <b>{HISTORICAL_TRANSACTIONS:,}</b>
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

                    These metrics provide a broader view of
                    discrimination and fraud-class performance
                    than accuracy alone.

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

    page_header(
        "Historical Model Analysis",
        "Historical transaction behaviour, fraud distribution and model evaluation.",
        "● HISTORICAL DATA",
        "info"
    )

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.metric(
            "Transactions",
            f"{HISTORICAL_TRANSACTIONS:,}"
        )

    with h2:
        st.metric(
            "Fraud Cases",
            f"{HISTORICAL_FRAUD:,}"
        )

    with h3:
        st.metric(
            "Fraud Rate",
            f"{HISTORICAL_FRAUD_RATE:.2f}%"
        )

    with h4:
        st.metric(
            "ROC-AUC",
            f"{MODEL_ROC_AUC:.2f}%"
        )

    st.markdown(
        '<div class="section-title">Historical Risk Profile</div>',
        unsafe_allow_html=True
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

    amount_chart = create_historical_amount_chart()

    if amount_chart is not None:

        st.markdown(
            '<div class="section-title">Transaction Amount Distribution</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(
            amount_chart,
            width="stretch"
        )

    if historical_df is not None:

        st.markdown(
            '<div class="section-title">Historical Data Observation</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            historical_df.head(20),
            width="stretch",
            height=400
        )

    st.markdown(
        """
        <div class="research-note">

        The historical analysis represents the offline analytical
        component of the framework. It provides the data foundation
        for understanding transaction behaviour and evaluating
        the machine learning model before operational deployment.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICT FRAUD
# ============================================================

elif page == "Predict Fraud":

    page_header(
        "Predict Fraud",
        "Submit a transaction feature vector for model-based fraud risk assessment.",
        "● MODEL READY",
        "live"
    )

    if model is None:

        st.error(
            "The Random Forest model could not be loaded. "
            "Ensure rf_fraud_model.pkl is in the project folder."
        )

    else:

        st.markdown(
            """
            <div class="research-note">

            Enter transaction information below. The model uses
            the same feature structure used during model training.
            Missing feature values are set to zero.

            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("fraud_prediction_form"):

            col1, col2, col3 = st.columns(3)

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

            with col3:

                threshold = st.slider(
                    "Decision Threshold",
                    0.01,
                    0.99,
                    0.50,
                    0.01
                )

            st.markdown(
                "### PCA Features"
            )

            feature_values = {}

            feature_cols = [
                f"V{i}"
                for i in range(1, 29)
            ]

            cols = st.columns(4)

            for index, feature in enumerate(
                feature_cols
            ):

                with cols[index % 4]:

                    feature_values[feature] = (
                        st.number_input(
                            feature,
                            value=0.0,
                            format="%.6f",
                            key=f"predict_{feature}"
                        )
                    )

            submitted = st.form_submit_button(
                "🔍 Predict Transaction Risk",
                use_container_width=True
            )

        if submitted:

            transaction = {
                "Time": transaction_time,
                "Amount": amount,
                **feature_values
            }

            try:

                X = prepare_transaction(
                    transaction
                )

                probability = (
                    model.predict_proba(X)[0][1]
                )

                probability_pct = (
                    probability * 100
                )

                prediction = (
                    "FRAUD"
                    if probability >= threshold
                    else "LEGITIMATE"
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
                                {probability_pct:.2f}%
                            </div>

                            <div>
                                Decision threshold:
                                {threshold * 100:.1f}%
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
                                ✓ LEGITIMATE RISK CLASSIFICATION
                            </div>

                            <div>
                                Estimated fraud probability
                            </div>

                            <div class="risk-probability">
                                {probability_pct:.2f}%
                            </div>

                            <div>
                                Decision threshold:
                                {threshold * 100:.1f}%
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# ============================================================
# REAL-TIME FRAUD DETECTION
# ============================================================

elif page == "Real-Time Fraud Detection":

    page_header(
        "Real-Time Fraud Detection",
        "Live transaction monitoring and classification from the Kafka event stream.",
        (
            "● KAFKA STREAM ACTIVE"
            if st.session_state.kafka_status == "Connected"
            else "● KAFKA OFFLINE — RETRYING"
        ),
        (
            "live"
            if st.session_state.kafka_status == "Connected"
            else "offline"
        )
    )

    # --------------------------------------------------------
    # ONLY THIS PAGE READS KAFKA
    # --------------------------------------------------------

    new_transactions = (
        read_kafka_transactions()
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

    last_probability = 0

    if st.session_state.last_transaction:

        last_probability = (
            st.session_state.last_transaction.get(
                "fraud_probability",
                0
            )
        )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Transactions Received
                </div>

                <div class="kpi-value">
                    {received_count}
                </div>

                <div class="kpi-description">
                    Kafka events received this session
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Fraud Detected
                </div>

                <div class="kpi-value">
                    {fraud_count}
                </div>

                <div class="kpi-description">
                    Transactions classified as fraud
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Legitimate
                </div>

                <div class="kpi-value">
                    {legitimate_count}
                </div>

                <div class="kpi-description">
                    Transactions classified as legitimate
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with k4:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Real-Time Fraud Rate
                </div>

                <div class="kpi-value">
                    {fraud_rate:.2f}%
                </div>

                <div class="kpi-description">
                    Fraud among received events
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with k5:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Latest Risk
                </div>

                <div class="kpi-value">
                    {last_probability:.2f}%
                </div>

                <div class="kpi-description">
                    Latest transaction fraud probability
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # RISK + LATEST
    # --------------------------------------------------------

    r1, r2 = st.columns(2)

    with r1:

        st.markdown(
            "### Real-Time Risk Distribution"
        )

        if received_count > 0:

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

            else:

                st.warning(
                    "Transaction received but could not be classified."
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

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Recent Transactions</div>',
        unsafe_allow_html=True
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
        "Kafka is polled only while this page is active. "
        "Switching to another page does not consume the Kafka stream."
    )

    if st.button(
        "🔄 Refresh Real-Time Data"
    ):

        st.rerun()


# ============================================================
# TRANSACTION EXPLORER
# ============================================================

elif page == "Transaction Explorer":

    page_header(
        "Transaction Explorer",
        "Explore historical transactions and, where available, live Kafka transactions.",
        "● EXPLORER",
        "info"
    )

    source = st.radio(
        "Transaction source",
        [
            "Historical Dataset",
            "Real-Time Session"
        ],
        horizontal=True
    )

    if source == "Historical Dataset":

        if historical_df is None:

            st.warning(
                "Historical dataset could not be loaded."
            )

        else:

            df = historical_df.copy()

            search_amount = st.number_input(
                "Minimum Amount",
                min_value=0.0,
                value=0.0
            )

            if "Amount" in df.columns:

                df = df[
                    df["Amount"] >= search_amount
                ]

            if "Class" in df.columns:

                class_filter = st.selectbox(
                    "Transaction Class",
                    [
                        "All",
                        "Legitimate",
                        "Fraud"
                    ]
                )

                if class_filter == "Fraud":

                    df = df[
                        df["Class"] == 1
                    ]

                elif class_filter == "Legitimate":

                    df = df[
                        df["Class"] == 0
                    ]

            st.write(
                f"Showing {len(df):,} transactions"
            )

            st.dataframe(
                df.head(1000),
                width="stretch",
                height=500
            )

    else:

        realtime_df = pd.DataFrame(
            list(
                st.session_state.realtime_transactions
            )
        )

        if realtime_df.empty:

            st.info(
                "No real-time transactions are currently stored."
            )

        else:

            st.dataframe(
                realtime_df,
                width="stretch",
                height=500
            )


# ============================================================
# TRENDS & MONITORING
# ============================================================

elif page == "Trends & Monitoring":

    page_header(
        "Trends & Monitoring",
        "Monitor transaction volumes, fraud rates and risk behaviour.",
        "● MONITORING",
        "info"
    )

    if historical_df is not None:

        if "Amount" in historical_df.columns:

            trend_df = historical_df.copy()

            trend_df["Amount_Band"] = pd.cut(
                trend_df["Amount"],
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
                trend_df.groupby(
                    "Amount_Band",
                    observed=False
                )
                .agg(
                    Transactions=(
                        "Amount",
                        "count"
                    ),
                    Average_Amount=(
                        "Amount",
                        "mean"
                    )
                )
                .reset_index()
            )

            fig = px.bar(
                grouped,
                x="Amount_Band",
                y="Transactions",
                labels={
                    "Amount_Band":
                        "Transaction Amount",
                    "Transactions":
                        "Transaction Count"
                }
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            st.dataframe(
                grouped,
                width="stretch",
                hide_index=True
            )

    else:

        st.info(
            "Historical dataset is not available."
        )

    realtime_count = len(
        st.session_state.realtime_transactions
    )

    realtime_fraud = sum(
        1
        for x in
        st.session_state.realtime_transactions
        if x.get("prediction") == "FRAUD"
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Session Transactions",
            realtime_count
        )

    with m2:

        st.metric(
            "Session Fraud",
            realtime_fraud
        )

    with m3:

        st.metric(
            "Session Fraud Rate",
            f"{(
                realtime_fraud /
                realtime_count *
                100
                if realtime_count
                else 0
            ):.2f}%"
        )


# ============================================================
# ADAPTIVE MONITORING
# ============================================================

elif page == "Adaptive Monitoring":

    page_header(
        "Adaptive Monitoring",
        "Monitor changing transaction behaviour and identify signals that may require model review.",
        "● ADAPTIVE LAYER",
        "info"
    )

    st.markdown(
        """
        <div class="research-note">

        Adaptive monitoring is a framework component rather than
        an assertion that automatic production retraining is
        currently occurring. The purpose is to identify changes
        in incoming data, risk behaviour or operational outcomes
        that may justify model reassessment.

        </div>
        """,
        unsafe_allow_html=True
    )

    transactions = list(
        st.session_state.realtime_transactions
    )

    if transactions:

        probabilities = [
            x.get(
                "fraud_probability",
                0
            )
            for x in transactions
        ]

        avg_risk = np.mean(
            probabilities
        )

        high_risk = sum(
            1
            for p in probabilities
            if p >= 50
        )

        a1, a2, a3 = st.columns(3)

        with a1:

            st.metric(
                "Average Observed Risk",
                f"{avg_risk:.2f}%"
            )

        with a2:

            st.metric(
                "High-Risk Events",
                high_risk
            )

        with a3:

            st.metric(
                "Events Monitored",
                len(transactions)
            )

        risk_df = pd.DataFrame(
            {
                "Event": range(
                    1,
                    len(probabilities) + 1
                ),
                "Fraud Probability":
                    probabilities
            }
        )

        fig = px.line(
            risk_df,
            x="Event",
            y="Fraud Probability",
            labels={
                "Fraud Probability":
                    "Fraud Probability (%)"
            }
        )

        fig.add_hline(
            y=50,
            line_dash="dash",
            annotation_text="50% reference"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.info(
            "No real-time observations are available yet. "
            "Open Real-Time Fraud Detection while Kafka is running."
        )


# ============================================================
# MODEL ADAPTATION
# ============================================================

elif page == "Model Adaptation":

    page_header(
        "Model Adaptation",
        "Conceptual pathway for maintaining model relevance as transaction behaviour changes.",
        "● ADAPTATION FRAMEWORK",
        "info"
    )

    steps = [
        (
            "Monitor",
            "Observe incoming transactions and relevant risk indicators."
        ),
        (
            "Detect Change",
            "Identify meaningful changes in transaction or fraud behaviour."
        ),
        (
            "Investigate",
            "Review data quality, model performance and operational feedback."
        ),
        (
            "Update",
            "Where justified, retrain or recalibrate the model using appropriate historical data."
        ),
        (
            "Validate",
            "Evaluate the updated model before considering operational use."
        ),
        (
            "Deploy",
            "Introduce an approved model version into the decision-support pipeline."
        )
    ]

    for i, (title, description) in enumerate(
        steps,
        1
    ):

        st.markdown(
            f"""
            <div class="workflow-step">

                <span class="workflow-number">
                    {i}
                </span>

                <span class="workflow-title">
                    {title}
                </span>

                <div class="workflow-description">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="research-note">

        The adaptation process is deliberately separated from
        real-time prediction. A live event should not automatically
        retrain a model without validation, governance and
        controlled model-management procedures.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# OPERATIONAL FEEDBACK
# ============================================================

elif page == "Operational Feedback":

    page_header(
        "Operational Feedback",
        "Capture human review outcomes that can support future evaluation and learning.",
        "● FEEDBACK",
        "info"
    )

    st.markdown(
        """
        Operational feedback provides a mechanism for a reviewer
        to record how a model decision was handled. This supports
        a future learning loop without assuming that every
        feedback record should immediately change the model.
        """
    )

    with st.form("operational_feedback"):

        transaction_id = st.text_input(
            "Transaction ID"
        )

        model_decision = st.selectbox(
            "Model Decision",
            [
                "FRAUD",
                "LEGITIMATE",
                "UNKNOWN"
            ]
        )

        operational_outcome = st.selectbox(
            "Operational Outcome",
            [
                "Confirmed Fraud",
                "Confirmed Legitimate",
                "False Positive",
                "False Negative",
                "Requires Further Review"
            ]
        )

        reviewer_comment = st.text_area(
            "Reviewer Comment"
        )

        submitted = st.form_submit_button(
            "Save Operational Feedback"
        )

    if submitted:

        record = {
            "transaction_id":
                transaction_id,

            "model_decision":
                model_decision,

            "operational_outcome":
                operational_outcome,

            "comment":
                reviewer_comment,

            "timestamp":
                pd.Timestamp.now()
        }

        st.session_state.feedback_records.append(
            record
        )

        st.success(
            "Operational feedback recorded for this session."
        )

    if st.session_state.feedback_records:

        st.markdown(
            "### Session Feedback"
        )

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

    page_header(
        "Feedback Learning Pipeline",
        "How operational outcomes can contribute to future model improvement.",
        "● LEARNING LOOP",
        "info"
    )

    learning_steps = [
        (
            "1",
            "Transaction Prediction",
            "The model generates a fraud-risk prediction."
        ),
        (
            "2",
            "Operational Review",
            "A human or operational process reviews selected cases."
        ),
        (
            "3",
            "Feedback Capture",
            "The outcome of the review is recorded."
        ),
        (
            "4",
            "Data Quality & Label Review",
            "Feedback is assessed before being considered suitable for learning."
        ),
        (
            "5",
            "Model Evaluation",
            "The candidate data can be used to evaluate model performance."
        ),
        (
            "6",
            "Controlled Adaptation",
            "An approved retraining or recalibration process can be performed."
        )
    ]

    for number, title, description in learning_steps:

        st.markdown(
            f"""
            <div class="workflow-step">

                <span class="workflow-number">
                    {number}
                </span>

                <span class="workflow-title">
                    {title}
                </span>

                <div class="workflow-description">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="research-note">

        This feedback loop is intended as a controlled learning
        pathway. Operational feedback should be validated before
        becoming training data to reduce the risk of propagating
        incorrect labels or operational bias.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# COST / VALUE FRAMING
# ============================================================

elif page == "Cost / Value Framing":

    page_header(
        "Cost / Value Framing",
        "Frame fraud decisions in terms of competing operational costs and potential value.",
        "● DECISION ECONOMICS",
        "info"
    )

    st.markdown(
        """
        Fraud detection is not only a classification problem.
        Operational systems must also consider the consequences
        of false positives and false negatives.

        A false negative may allow fraudulent activity to proceed,
        while a false positive may impose unnecessary friction
        on a legitimate customer or transaction.
        """
    )

    c1, c2 = st.columns(2)

    with c1:

        false_negative_cost = st.number_input(
            "Illustrative False-Negative Cost",
            min_value=0.0,
            value=1000.0
        )

        false_positive_cost = st.number_input(
            "Illustrative False-Positive Cost",
            min_value=0.0,
            value=20.0
        )

    with c2:

        expected_false_negatives = st.number_input(
            "Expected False Negatives",
            min_value=0,
            value=10
        )

        expected_false_positives = st.number_input(
            "Expected False Positives",
            min_value=0,
            value=50
        )

    estimated_cost = (
        false_negative_cost *
        expected_false_negatives
        +
        false_positive_cost *
        expected_false_positives
    )

    st.metric(
        "Illustrative Operational Cost",
        f"{estimated_cost:,.2f}"
    )

    st.markdown(
        """
        <div class="research-note">

        The values above are illustrative rather than empirical
        banking cost estimates. They demonstrate how threshold
        selection can be connected to operational decision-making.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# THRESHOLD TUNER
# ============================================================

elif page == "Threshold Tuner":

    page_header(
        "Threshold Tuner",
        "Explore how the fraud decision threshold affects classification behaviour.",
        "● DECISION SUPPORT",
        "info"
    )

    threshold = st.slider(
        "Fraud Decision Threshold",
        0.01,
        0.99,
        st.session_state.threshold_value,
        0.01
    )

    st.session_state.threshold_value = threshold

    st.metric(
        "Selected Threshold",
        f"{threshold * 100:.1f}%"
    )

    st.markdown(
        """
        A lower threshold generally increases sensitivity to
        potential fraud but may also increase the number of
        legitimate transactions requiring review. A higher
        threshold may reduce unnecessary alerts while allowing
        some higher-risk events to remain below the decision
        boundary.

        The appropriate threshold should therefore be determined
        using validation data and operational cost considerations.
        """
    )

    if (
        st.session_state.realtime_transactions
    ):

        probabilities = [
            x.get(
                "fraud_probability",
                0
            ) / 100
            for x in
            st.session_state.realtime_transactions
        ]

        classified_fraud = sum(
            p >= threshold
            for p in probabilities
        )

        st.metric(
            "Current Session Events Above Threshold",
            classified_fraud
        )

    else:

        st.info(
            "No live session transactions are currently available."
        )


# ============================================================
# AI DECISION ASSISTANT
# ============================================================

elif page == "AI Decision Assistant":

    page_header(
        "AI Decision Assistant",
        "Interpret model outputs and translate risk information into operationally useful guidance.",
        "● ASSISTIVE LAYER",
        "info"
    )

    latest = (
        st.session_state.last_transaction
    )

    if latest is None:

        st.info(
            "No latest real-time transaction is available."
        )

        st.markdown(
            """
            The assistant can be used after transactions have
            been received through the real-time monitoring layer.
            """
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

        st.metric(
            "Latest Fraud Probability",
            f"{probability:.2f}%"
        )

        if prediction == "FRAUD":

            recommendation = (
                "The latest transaction has been classified as "
                "fraudulent by the current model threshold. "
                "A risk-based operational review is recommended."
            )

        elif prediction == "LEGITIMATE":

            recommendation = (
                "The latest transaction is currently classified "
                "as legitimate. No elevated model risk signal is "
                "present under the current decision threshold."
            )

        else:

            recommendation = (
                "The transaction could not be classified reliably. "
                "Operational review may be required."
            )

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-title">
                    Decision Support Interpretation
                </div>

                <div class="insight-text">
                    {recommendation}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="research-note">

            The assistant is a decision-support component. It does
            not independently establish the truth of a transaction
            and should not replace appropriate human or institutional
            controls.

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# END-TO-END DSS WORKFLOW
# ============================================================

elif page == "End-to-End DSS Workflow":

    page_header(
        "End-to-End DSS Workflow",
        "Integrated view of the proposed fraud detection and operational decision-support framework.",
        "● FRAMEWORK",
        "info"
    )

    workflow = [
        (
            "1",
            "Historical Data",
            "Historical transactions provide the analytical foundation for exploration, preprocessing and model development."
        ),
        (
            "2",
            "Model Development",
            "Machine learning models are trained and evaluated using appropriate validation procedures."
        ),
        (
            "3",
            "Transaction Ingestion",
            "Incoming events enter the operational pipeline through an event-streaming mechanism such as Kafka."
        ),
        (
            "4",
            "Real-Time Classification",
            "The trained model evaluates incoming transaction features and produces a fraud-risk probability."
        ),
        (
            "5",
            "Explainability",
            "Feature importance and explainability methods provide insight into model behaviour."
        ),
        (
            "6",
            "Decision Support",
            "Risk outputs are presented in an operational dashboard to support review and prioritisation."
        ),
        (
            "7",
            "Operational Feedback",
            "Review outcomes can be recorded and incorporated into a controlled learning pathway."
        ),
        (
            "8",
            "Adaptive Monitoring",
            "Changes in transaction behaviour and model performance can trigger investigation and potential adaptation."
        )
    ]

    for number, title, description in workflow:

        st.markdown(
            f"""
            <div class="workflow-step">

                <span class="workflow-number">
                    {number}
                </span>

                <span class="workflow-title">
                    {title}
                </span>

                <div class="workflow-description">
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

    page_header(
        "About the Framework",
        "Research framing, methodology and implementation scope.",
        "● RESEARCH FRAMEWORK",
        "info"
    )

    st.markdown(
        """
        <div class="dashboard-panel">

            <div class="panel-title">
                Adaptive and Interpretable Fraud Detection Framework
            </div>

            <div class="insight-text">

            The framework is designed to investigate how machine
            learning, explainability, real-time event streaming,
            adaptive monitoring and operational decision support
            can be integrated into a fraud detection environment.

            <br><br>

            The objective is not simply to classify transactions,
            but to provide a pathway through which model outputs
            can become interpretable and operationally useful.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Methodological Foundation</div>',
        unsafe_allow_html=True
    )

    method1, method2, method3 = st.columns(3)

    with method1:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-title">
                    CRISP-DM
                </div>

                <div class="insight-text">
                    Provides the analytical structure for data
                    understanding, preparation, modelling and
                    evaluation.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with method2:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-title">
                    MLOps Principles
                </div>

                <div class="insight-text">
                    Supports controlled model lifecycle,
                    monitoring, evaluation and potential
                    adaptation.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with method3:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-title">
                    Agile Prototyping
                </div>

                <div class="insight-text">
                    Supports iterative development and integration
                    of the analytical and operational components.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Technology Components</div>',
        unsafe_allow_html=True
    )

    technology_df = pd.DataFrame(
        {
            "Component": [
                "Python",
                "Pandas / NumPy",
                "Scikit-learn",
                "Random Forest",
                "Apache Kafka",
                "Streamlit",
                "Plotly",
                "SHAP / Feature Importance"
            ],

            "Role": [
                "Core implementation",
                "Data preparation and analysis",
                "Machine learning framework",
                "Fraud classification",
                "Real-time event streaming",
                "Decision-support interface",
                "Interactive visualisation",
                "Model interpretation"
            ]
        }
    )

    st.dataframe(
        technology_df,
        width="stretch",
        hide_index=True
    )

    st.markdown(
        """
        <div class="research-note">

        <b>Implementation scope:</b> This system is a prototype /
        reference implementation. Synthetic transactions are used
        to demonstrate the real-time streaming component. In a
        production banking environment, secure enterprise
        transaction sources and governed APIs or event-streaming
        infrastructure would replace the synthetic generator.

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