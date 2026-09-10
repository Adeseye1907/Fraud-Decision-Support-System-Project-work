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
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION & PATHS
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
# MODEL BENCHMARK METRICS
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
    "About the Framework",
]


# ============================================================
# CUSTOM CSS (EMERALD & DARK-SLATE THEME)
# ============================================================

st.markdown(
    """
    <style>
    /* Global Layout */
    .stApp {
        background: #f5f7fb;
    }

    .main {
        padding-top: 1rem;
    }

    /* Force crisp dark headers */
    h1, h2, h3, h4, .stSubheader, [data-testid="stSubheader"], [data-testid="stMarkdownContainer"] h3 {
        color: #10233f !important;
        font-weight: 750 !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
        background: transparent !important;
    }

    /* Description and body text */
    p, span, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, [data-testid="stMarkdownContainer"] p {
        color: #475569 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
    }

    /* Labels */
    [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p, label {
        color: #1e293b !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071b33 0%, #0b2747 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .sidebar-brand {
        font-size: 24px;
        font-weight: 800;
        color: white !important;
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

    /* Headers & Badges */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
    }

    .dashboard-title {
        font-size: 30px;
        font-weight: 800;
        color: #10233f !important;
        line-height: 1.15;
        margin-bottom: 5px;
    }

    .dashboard-subtitle {
        color: #667085 !important;
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

    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 20px !important;
        margin-right: 6px !important;
    }

    button[data-baseweb="tab"] div p,
    button[data-baseweb="tab"] p {
        color: #475569 !important;
        font-size: 14.5px !important;
        font-weight: 600 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #065f46 !important;
        border-color: #065f46 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] div p,
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #059669 !important;
        height: 3px !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #059669 !important;
        color: #ffffff !important;
        border: 1px solid #047857 !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }

    div.stButton > button p,
    div.stButton > button div,
    div.stButton > button span {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    div.stButton > button:hover {
        background-color: #047857 !important;
        border-color: #065f46 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(5, 150, 105, 0.35) !important;
    }

    /* Cards */
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
        color: #172b4d !important;
        margin-bottom: 8px;
    }

    .panel-subtitle {
        font-size: 12px;
        color: #8a94a6 !important;
        margin-bottom: 15px;
    }

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
        color: #6b7280 !important;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .kpi-value {
        font-size: 27px;
        font-weight: 800;
        color: #12233f !important;
        margin-bottom: 5px;
    }

    .kpi-description {
        font-size: 11px;
        color: #8b95a7 !important;
    }

    .kpi-icon {
        float: right;
        font-size: 20px;
    }

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
        color: #14243d !important;
        margin-bottom: 8px;
    }

    .insight-text {
        font-size: 13px;
        line-height: 1.65;
        color: #667085 !important;
    }

    .footer {
        margin-top: 35px;
        padding: 20px 0;
        text-align: center;
        color: #98a2b3 !important;
        font-size: 11px;
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


# ============================================================
# LOAD MODEL & SCALER
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
# LOAD HISTORICAL DATA (RESILIENT PATH RESOLUTION)
# ============================================================

@st.cache_data(show_spinner=False)
def load_historical_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    candidate_paths = [
        os.path.join(base_dir, "creditcard.csv"),
        os.path.join(base_dir, "data", "creditcard.csv"),
        os.path.join(base_dir, "historical_predictions.csv"),
        os.path.join(base_dir, "historical_test_data.csv"),
        "creditcard.csv",
        "data/creditcard.csv",
        "historical_predictions.csv",
        "historical_test_data.csv",
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df
            except Exception:
                continue
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
# SESSION STATE INITIALIZATION
# ============================================================

if "realtime_transactions" not in st.session_state:
    st.session_state.realtime_transactions = deque(maxlen=MAX_REALTIME_TRANSACTIONS)

if "transactions" not in st.session_state:
    st.session_state.transactions = []

if "realtime_ids" not in st.session_state:
    st.session_state.realtime_ids = set()

if "kafka_status" not in st.session_state:
    st.session_state.kafka_status = "Not connected"

if "last_transaction" not in st.session_state:
    st.session_state.last_transaction = None

if "feedback_records" not in st.session_state:
    st.session_state.feedback_records = []

if "adaptation_history" not in st.session_state:
    st.session_state.adaptation_history = []

if "last_drift_status" not in st.session_state:
    st.session_state.last_drift_status = "Not evaluated"

if "last_drift_score" not in st.session_state:
    st.session_state.last_drift_score = 0.0


# ============================================================
# KAFKA CONSUMER & INGESTION
# ============================================================

@st.cache_resource(show_spinner=False)
def create_kafka_consumer():
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="fraud-dashboard-consumer-v5",
        request_timeout_ms=3000,
        session_timeout_ms=6000,
        heartbeat_interval_ms=2000,
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
                transaction["transaction_id"] = transaction_id

                st.session_state.realtime_transactions.append(transaction)
                st.session_state.transactions.append(transaction)
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


read_kafka_transactions()


# ============================================================
# CHART HELPER FUNCTIONS (DARK SLATE & SKY-BLUE THEME)
# ============================================================

def create_donut_chart(fraud, legitimate, title=""):
    total = fraud + legitimate
    fraud_pct = (fraud / total * 100) if total > 0 else 0.0

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Legitimate", "Fraud"],
                values=[legitimate, fraud],
                hole=0.68,
                textinfo="percent",
                textposition="outside",
                marker=dict(
                    colors=["#7dc4fc", "#0056b3"],
                    line=dict(color="#111116", width=2)
                ),
                textfont=dict(color="#ffffff", size=13),
                insidetextfont=dict(color="#ffffff", size=13),
                outsidetextfont=dict(color="#ffffff", size=13),
                hovertemplate="<b>%{label}</b><br>Transactions: %{value:,}<br>Share: %{percent}<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=30, b=30),
        showlegend=True,
        paper_bgcolor="#111116",
        plot_bgcolor="#111116",
        font=dict(color="#ffffff", size=12),
        legend=dict(
            orientation="h",
            y=-0.12,
            x=0.5,
            xanchor="center",
            font=dict(color="#ffffff", size=13)
        ),
        annotations=[
            dict(
                text=f"<b style='font-size:20px; color:#ffffff;'>{fraud_pct:.2f}%</b><br><span style='font-size:12px; color:#9ca3af;'>Fraud Rate</span>",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        ]
    )
    return fig


def create_model_performance_chart():
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    values = [MODEL_ACCURACY, MODEL_PRECISION, MODEL_RECALL, MODEL_F1, MODEL_ROC_AUC, MODEL_PR_AUC]
    display_values = [v * 100 if v <= 1.0 else v for v in values]

    fig = go.Figure(
        data=[
            go.Bar(
                x=metrics,
                y=display_values,
                text=[f"<b>{v:.2f}%</b>" for v in display_values],
                textposition="outside",
                textfont=dict(color="#ffffff", size=12),
                marker=dict(
                    color="#7dc4fc",
                    line=dict(color="#58a6eb", width=1.5),
                    cornerradius=4
                ),
                hovertemplate="<b>%{x}</b>: %{y:.2f}%<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=35, b=45),
        paper_bgcolor="#111116",
        plot_bgcolor="#111116",
        font=dict(color="#ffffff", size=12),
        xaxis=dict(
            title=dict(text="Metric", font=dict(color="#ffffff", size=13)),
            tickfont=dict(color="#ffffff", size=12),
            showgrid=False,
            tickangle=-35,
        ),
        yaxis=dict(
            title=dict(text="Percentage", font=dict(color="#ffffff", size=13)),
            range=[0, 115],
            tickfont=dict(color="#ffffff", size=12),
            gridcolor="#262633",
            showgrid=True,
        ),
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
        labels=["< $10", "$10–50", "$50–100", "$100–500", "$500–1K", "> $1K"],
    )
    grouped = (
        temp.groupby("Amount_Band", observed=False)
        .size()
        .reset_index(name="Transactions")
    )
    fig = px.bar(grouped, x="Amount_Band", y="Transactions", color_discrete_sequence=["#7dc4fc"])
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=30, b=30),
        paper_bgcolor="#111116",
        plot_bgcolor="#111116",
        font=dict(color="#ffffff", size=12),
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
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-section">
            SYSTEM NAVIGATION
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio("System Navigation", PAGES, label_visibility="collapsed")
    current_page = page

    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-author">
            Prototype / Reference Implementation<br>
            Adaptive and interpretable fraud detection and operational decision support.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()


# ============================================================
# 1. INTRODUCTION (ROUTING STARTS HERE)
# ============================================================

if page == "Introduction":
    # Introduction page logic continues...