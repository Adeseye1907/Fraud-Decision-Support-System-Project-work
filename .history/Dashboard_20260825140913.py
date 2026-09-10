# ============================================================
# DASHBOARD.PY
# FRAUD DETECTION & OPERATIONAL DECISION SUPPORT SYSTEM
# ============================================================

import os
import json
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
    page_title="Fraud Detection DSS",
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
# HISTORICAL MODEL METRICS
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
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #b8c8dc !important;
        margin-bottom: 30px;
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
        margin-bottom: 2px;
    }

    .dashboard-subtitle {
        color: #6b7280;
        font-size: 14px;
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
       KPI CARDS
       ====================================================== */

    .kpi-card {
        background: white;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #e8edf4;
        box-shadow:
            0 4px 16px rgba(15, 23, 42, 0.04);
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
        line-height: 1.5;
    }

    .kpi-icon {
        float: right;
        font-size: 20px;
    }


    /* ======================================================
       SECTION HEADINGS
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
        line-height: 1.6;
    }


    /* ======================================================
       PANELS
       ====================================================== */

    .dashboard-panel {
        background: white;
        border: 1px solid #e8edf4;
        border-radius: 14px;
        padding: 18px;
        box-shadow:
            0 4px 16px rgba(15, 23, 42, 0.035);
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
        box-shadow:
            0 4px 16px rgba(15, 23, 42, 0.025);
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
       REAL-TIME TRANSACTION CARD
       ====================================================== */

    .transaction-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e8edf4;
        box-shadow:
            0 4px 18px rgba(15, 23, 42, 0.05);
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
       SMALL SCREEN ADJUSTMENTS
       ====================================================== */

    @media (max-width: 900px) {

        .dashboard-title {
            font-size: 24px;
        }

        .dashboard-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
        }

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


# ============================================================
# LOAD SCALER
# ============================================================

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

        df = pd.read_csv(DATA_PATH)

        return df

    except Exception:

        return None


historical_df = load_historical_data()


# ============================================================
# HISTORICAL DATA STATISTICS
# ============================================================

if historical_df is not None:

    HISTORICAL_TRANSACTIONS = len(
        historical_df
    )

    if "Class" in historical_df.columns:

        HISTORICAL_FRAUD = int(
            historical_df["Class"].sum()
        )

    else:

        HISTORICAL_FRAUD = 492

    HISTORICAL_LEGITIMATE = (
        HISTORICAL_TRANSACTIONS
        - HISTORICAL_FRAUD
    )

else:

    HISTORICAL_TRANSACTIONS = 284807
    HISTORICAL_FRAUD = 492

    HISTORICAL_LEGITIMATE = (
        HISTORICAL_TRANSACTIONS
        - HISTORICAL_FRAUD
    )


HISTORICAL_FRAUD_RATE = (
    HISTORICAL_FRAUD
    / HISTORICAL_TRANSACTIONS
    * 100
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

        bootstrap_servers=[
            KAFKA_SERVER
        ],

        value_deserializer=lambda x:
            json.loads(
                x.decode("utf-8")
            ),

        auto_offset_reset="latest",

        enable_auto_commit=True,

        group_id="fraud-dashboard-consumer-v2",

        request_timeout_ms=3000,

        session_timeout_ms=6000,

        heartbeat_interval_ms=2000,

        consumer_timeout_ms=100

    )

    return consumer


# ============================================================
# CLOSE KAFKA CONSUMER
# ============================================================

def close_kafka_consumer():

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
# MODEL FEATURE PREPARATION
# ============================================================

DEFAULT_FEATURES = (
    ["Time", "Amount"]
    + [f"V{i}" for i in range(1, 29)]
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

    """
    Converts a Kafka transaction into a DataFrame
    with exactly the feature names expected by the
    Random Forest model.
    """

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


    # --------------------------------------------------------
    # Scale Time and Amount
    # --------------------------------------------------------

    cols_to_scale = [
        column
        for column in [
            "Time",
            "Amount"
        ]
        if column in X.columns
    ]


    if scaler is not None and cols_to_scale:

        try:

            # The project scaler was trained on
            # Time and Amount.

            X[cols_to_scale] = scaler.transform(
                X[cols_to_scale]
            )

        except Exception:

            # If the saved scaler has a different
            # expected structure, keep the original
            # values rather than crashing.

            pass


    return X


# ============================================================
# REAL-TIME PREDICTION
# ============================================================

def predict_transaction(transaction):

    transaction = dict(
        transaction
    )


    if model is None:

        transaction["prediction"] = "UNKNOWN"

        transaction[
            "fraud_probability"
        ] = 0.0

        return transaction


    try:

        X = prepare_transaction(
            transaction
        )


        prediction = model.predict(
            X
        )[0]


        probability = model.predict_proba(
            X
        )[0][1]


        transaction["prediction"] = (

            "FRAUD"

            if int(prediction) == 1

            else "LEGITIMATE"

        )


        transaction[
            "fraud_probability"
        ] = (
            float(probability) * 100
        )


    except Exception as e:

        transaction["prediction"] = "ERROR"

        transaction[
            "fraud_probability"
        ] = 0.0

        transaction[
            "_prediction_error"
        ] = str(e)


    return transaction


# ============================================================
# READ KAFKA TRANSACTIONS
# ============================================================

def read_kafka_transactions():

    """
    Kafka is intentionally read only from the
    Real-Time Monitoring page.

    The Executive Overview never creates or polls
    a Kafka consumer.
    """

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


                # ------------------------------------------------
                # CREATE TRANSACTION ID
                # ------------------------------------------------

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


                # ------------------------------------------------
                # DUPLICATE CHECK
                # ------------------------------------------------

                if (
                    transaction_id
                    in st.session_state.realtime_ids
                ):

                    continue


                st.session_state.realtime_ids.add(
                    transaction_id
                )


                # ------------------------------------------------
                # PREDICT
                # ------------------------------------------------

                transaction = predict_transaction(
                    transaction
                )


                transaction[
                    "_transaction_id"
                ] = transaction_id


                # ------------------------------------------------
                # BOUNDED MEMORY
                # ------------------------------------------------

                st.session_state.realtime_transactions.append(
                    transaction
                )


                new_transactions.append(
                    transaction
                )


        # --------------------------------------------------------
        # PREVENT ID SET FROM GROWING FOREVER
        # --------------------------------------------------------

        if len(
            st.session_state.realtime_ids
        ) > (
            MAX_REALTIME_TRANSACTIONS * 2
        ):

            st.session_state.realtime_ids = {

                str(
                    x.get(
                        "_transaction_id"
                    )
                )

                for x
                in st.session_state.realtime_transactions

                if x.get(
                    "_transaction_id"
                ) is not None

            }


        # --------------------------------------------------------
        # STORE LATEST TRANSACTION
        # --------------------------------------------------------

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


def format_percentage(value):

    return f"{value:.2f}%"


# ============================================================
# DONUT CHART
# ============================================================

def create_donut_chart(
    fraud,
    legitimate,
    title
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
            t=50,
            b=10
        ),

        showlegend=True,

        legend=dict(

            orientation="h",

            y=-0.05

        )

    )


    return fig


# ============================================================
# MODEL PERFORMANCE CHART
# ============================================================

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

        range=[
            0,
            110
        ]

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


# ============================================================
# HISTORICAL AMOUNT CHART
# ============================================================

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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            🛡️ FraudGuard DSS
        </div>

        <div class="sidebar-subtitle">
            Adaptive & Interpretable Fraud Detection
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "### Navigation"
    )


    page = st.radio(

        "Dashboard",

        [

            "Executive Overview",

            "Historical Analysis",

            "Real-Time Monitoring",

            "Explainability",

            "System Monitoring"

        ],

        label_visibility="collapsed"

    )


    st.markdown("---")


    st.markdown(
        """
        **System Architecture**

        Historical Data  
        ↓  
        Random Forest  
        ↓  
        Kafka Streaming  
        ↓  
        Real-Time Classification  
        ↓  
        XAI  
        ↓  
        Decision Support
        """
    )


    st.markdown("---")


    st.caption(
        "Prototype / reference implementation for "
        "adaptive and interpretable fraud detection."
    )


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="dashboard-header">

            <div>

                <div class="dashboard-title">
                    Executive Dashboard
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
    # EXECUTIVE CONTEXT
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="dashboard-panel"
             style="
                margin-bottom:20px;
                background:linear-gradient(
                    135deg,
                    #ffffff 0%,
                    #f8fafc 100%
                );
             ">

            <div class="panel-title">
                Executive Overview
            </div>

            <div class="panel-subtitle"
                 style="
                    font-size:13px;
                    line-height:1.7;
                    margin-bottom:0;
                 ">

                This dashboard provides a high-level view of
                fraud exposure, model performance and the
                operational decision-support pipeline.

                Historical indicators summarise the evaluated
                transaction dataset, while system readiness
                reflects the availability of the supporting
                detection components.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # KPI CARDS
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)


    # --------------------------------------------------------
    # TRANSACTIONS
    # --------------------------------------------------------

    with c1:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-icon">
                    📊
                </div>

                <div class="kpi-label">
                    Transactions
                </div>

                <div class="kpi-value">
                    {format_number(
                        HISTORICAL_TRANSACTIONS
                    )}
                </div>

                <div class="kpi-description">
                    Historical transactions analysed
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # FRAUD CASES
    # --------------------------------------------------------

    with c2:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-icon">
                    🚨
                </div>

                <div class="kpi-label">
                    Fraud Cases
                </div>

                <div class="kpi-value">
                    {format_number(
                        HISTORICAL_FRAUD
                    )}
                </div>

                <div class="kpi-description">
                    Identified fraudulent transactions
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # FRAUD RATE
    # --------------------------------------------------------

    with c3:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-icon">
                    ⚠️
                </div>

                <div class="kpi-label">
                    Fraud Rate
                </div>

                <div class="kpi-value">
                    {HISTORICAL_FRAUD_RATE:.2f}%
                </div>

                <div class="kpi-description">
                    Fraud proportion in dataset
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    with c4:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-icon">
                    🎯
                </div>

                <div class="kpi-label">
                    ROC-AUC
                </div>

                <div class="kpi-value">
                    {MODEL_ROC_AUC:.2f}%
                </div>

                <div class="kpi-description">
                    Model discrimination capability
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    with c5:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-icon">
                    ⚖️
                </div>

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


    # ========================================================
    # RISK & MODEL PERFORMANCE
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
            Risk & Model Performance
        </div>

        <div class="section-subtitle">
            Overview of historical transaction risk and the
            evaluated fraud-detection model.
        </div>
        """,
        unsafe_allow_html=True
    )


    chart1, chart2 = st.columns([1, 1])


    # --------------------------------------------------------
    # DONUT
    # --------------------------------------------------------

    with chart1:

        st.markdown(
            """
            <div class="dashboard-panel">

                <div class="panel-title">
                    Transaction Risk Distribution
                </div>

                <div class="panel-subtitle">
                    Historical distribution of legitimate and
                    fraudulent transactions
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        risk_fig = create_donut_chart(

            HISTORICAL_FRAUD,

            HISTORICAL_LEGITIMATE,

            ""

        )


        risk_fig.update_layout(

            height=330,

            margin=dict(
                l=10,
                r=10,
                t=15,
                b=15
            )

        )


        st.plotly_chart(

            risk_fig,

            width="stretch",

            config={
                "displayModeBar": False
            }

        )


    # --------------------------------------------------------
    # PERFORMANCE CHART
    # --------------------------------------------------------

    with chart2:

        st.markdown(
            """
            <div class="dashboard-panel">

                <div class="panel-title">
                    Fraud Detection Model Performance
                </div>

                <div class="panel-subtitle">
                    Evaluation metrics from the Random Forest
                    fraud-detection model
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        performance_fig = (
            create_model_performance_chart()
        )


        performance_fig.update_layout(

            height=330,

            margin=dict(
                l=10,
                r=10,
                t=15,
                b=15
            )

        )


        st.plotly_chart(

            performance_fig,

            width="stretch",

            config={
                "displayModeBar": False
            }

        )


    # ========================================================
    # MODEL EVALUATION + DECISION SUPPORT
    # ========================================================

    left, right = st.columns([1, 1])


    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    with left:

        st.markdown(
            """
            <div class="section-title">
                Model Evaluation Summary
            </div>

            <div class="section-subtitle">
                Key performance indicators from the evaluated
                Random Forest model.
            </div>
            """,
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


    # --------------------------------------------------------
    # DECISION SUPPORT PIPELINE
    # --------------------------------------------------------

    with right:

        st.markdown(
            """
            <div class="section-title">
                Decision Support Pipeline
            </div>

            <div class="section-subtitle">
                Availability of the major components supporting
                the fraud-detection workflow.
            </div>
            """,
            unsafe_allow_html=True
        )


        model_ready = model is not None

        scaler_ready = scaler is not None

        data_ready = historical_df is not None


        model_status = (
            "READY"
            if model_ready
            else "UNAVAILABLE"
        )

        model_class = (
            "status-good"
            if model_ready
            else "status-danger"
        )


        scaler_status = (
            "READY"
            if scaler_ready
            else "NOT LOADED"
        )

        scaler_class = (
            "status-good"
            if scaler_ready
            else "status-warning"
        )


        data_status = (
            "AVAILABLE"
            if data_ready
            else "UNAVAILABLE"
        )

        data_class = (
            "status-good"
            if data_ready
            else "status-danger"
        )


        st.markdown(
            f"""
            <div class="dashboard-panel">

                <div class="status-row">

                    <span>
                        Historical Transaction Data
                    </span>

                    <span class="{data_class}">
                        ● {data_status}
                    </span>

                </div>

                <div class="status-row">

                    <span>
                        Random Forest Model
                    </span>

                    <span class="{model_class}">
                        ● {model_status}
                    </span>

                </div>

                <div class="status-row">

                    <span>
                        Feature Scaling
                    </span>

                    <span class="{scaler_class}">
                        ● {scaler_status}
                    </span>

                </div>

                <div class="status-row">

                    <span>
                        Kafka Event Streaming
                    </span>

                    <span class="status-good">
                        ● CONFIGURED
                    </span>

                </div>

                <div class="status-row">

                    <span>
                        Real-Time Classification
                    </span>

                    <span class="{model_class}">
                        ● {
                            "READY"
                            if model_ready
                            else "UNAVAILABLE"
                        }
                    </span>

                </div>

                <div class="status-row">

                    <span>
                        Explainability Layer
                    </span>

                    <span class="status-good">
                        ● READY
                    </span>

                </div>

                <div class="status-row">

                    <span>
                        Decision Support Dashboard
                    </span>

                    <span class="status-good">
                        ● ACTIVE
                    </span>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # EXECUTIVE INSIGHTS
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
            Executive Insights
        </div>

        <div class="section-subtitle">
            Key observations supporting operational
            interpretation of the fraud-detection results.
        </div>
        """,
        unsafe_allow_html=True
    )


    i1, i2, i3 = st.columns(3)


    # --------------------------------------------------------
    # FRAUD EXPOSURE
    # --------------------------------------------------------

    with i1:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-title">
                    🚨 Fraud Exposure
                </div>

                <div class="insight-text">

                    The evaluated historical dataset contains
                    <b>{HISTORICAL_FRAUD:,}</b> identified
                    fraudulent transactions out of
                    <b>{HISTORICAL_TRANSACTIONS:,}</b>
                    transactions analysed.

                    <br><br>

                    This corresponds to a historical fraud rate
                    of <b>{HISTORICAL_FRAUD_RATE:.2f}%</b>,
                    highlighting the highly imbalanced nature
                    of fraudulent activity.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # DETECTION CAPABILITY
    # --------------------------------------------------------

    with i2:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-title">
                    🎯 Detection Capability
                </div>

                <div class="insight-text">

                    The Random Forest model achieved a ROC-AUC
                    of <b>{MODEL_ROC_AUC:.2f}%</b> and a PR-AUC
                    of <b>{MODEL_PR_AUC:.2f}%</b> on the evaluated
                    test data.

                    <br><br>

                    These measures provide evidence of the model's
                    ability to distinguish fraudulent activity
                    from legitimate transactions.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # OPERATIONAL CONSIDERATION
    # --------------------------------------------------------

    with i3:

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-title">
                    ⚖️ Operational Consideration
                </div>

                <div class="insight-text">

                    Fraud recall is currently
                    <b>{MODEL_RECALL:.1f}%</b>, while precision
                    is <b>{MODEL_PRECISION:.1f}%</b>.

                    <br><br>

                    This supports the use of risk-based review,
                    where model outputs can assist analysts in
                    prioritising potentially suspicious
                    transactions rather than treating all
                    transactions equally.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # DECISION-SUPPORT INTERPRETATION
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
            What This Means for Decision Support
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="dashboard-panel">

            <div class="insight-text"
                 style="
                    font-size:13px;
                    line-height:1.8;
                 ">

                The proposed decision-support framework combines
                machine-learning classification with operational
                interpretation.

                The model provides a probability-based assessment
                of transaction risk, while the dashboard provides
                a human-readable view of the resulting risk
                indicators.

                <br><br>

                The current evaluation indicates strong overall
                discrimination, with a ROC-AUC of
                <b>{MODEL_ROC_AUC:.2f}%</b> and PR-AUC of
                <b>{MODEL_PR_AUC:.2f}%</b>. However, the recall of
                <b>{MODEL_RECALL:.1f}%</b> demonstrates that some
                fraudulent transactions may remain undetected.

                <br><br>

                Therefore, the framework is intended as a
                <b>decision-support mechanism</b> rather than an
                autonomous replacement for human review.

                Its operational value comes from combining
                detection, risk prioritisation, explainability
                and real-time event processing within a single
                interface.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # IMPLEMENTATION CONTEXT
    # ========================================================

    st.markdown(
        """
        <div class="dashboard-panel"
             style="
                margin-top:20px;
                background:#f8fafc;
             ">

            <div class="panel-title">
                Implementation Context
            </div>

            <div class="insight-text"
                 style="
                    font-size:12px;
                    line-height:1.7;
                 ">

                This dashboard represents a prototype/reference
                implementation of an adaptive and interpretable
                fraud-detection and operational decision-support
                framework.

                <br><br>

                The historical results shown above are based on
                the evaluated dataset and should not be interpreted
                as evidence of production banking deployment.
                In a real banking environment, secure enterprise
                transaction sources and governed event-streaming
                infrastructure would replace the prototype data
                sources.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """
        <div class="footer">

            Fraud Detection & Operational Decision Support System
            • Prototype / Reference Implementation

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================

elif page == "Historical Analysis":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="dashboard-header">

            <div>

                <div class="dashboard-title">
                    Historical Model Analysis
                </div>

                <div class="dashboard-subtitle">
                    Analysis of historical transaction behaviour
                    and model performance
                </div>

            </div>

            <div class="live-badge">
                ● HISTORICAL DATA
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    c1, c2 = st.columns(2)


    with c1:

        st.markdown(
            "### Transaction Risk Distribution"
        )


        st.plotly_chart(

            create_donut_chart(

                HISTORICAL_FRAUD,

                HISTORICAL_LEGITIMATE,

                ""

            ),

            width="stretch",

            config={
                "displayModeBar": False
            }

        )


    with c2:

        st.markdown(
            "### Model Evaluation"
        )


        st.plotly_chart(

            create_model_performance_chart(),

            width="stretch",

            config={
                "displayModeBar": False
            }

        )


    # --------------------------------------------------------
    # AMOUNT DISTRIBUTION
    # --------------------------------------------------------

    amount_chart = (
        create_historical_amount_chart()
    )


    if amount_chart is not None:

        st.markdown(
            "### Transaction Amount Distribution"
        )


        st.plotly_chart(

            amount_chart,

            width="stretch",

            config={
                "displayModeBar": False
            }

        )


    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    if historical_df is not None:

        st.markdown(
            "### Data Observation"
        )


        st.dataframe(

            historical_df.head(20),

            width="stretch",

            height=400,

            hide_index=True

        )


    else:

        st.warning(
            "Historical dataset could not be loaded."
        )


# ============================================================
# REAL-TIME MONITORING
# ============================================================

elif page == "Real-Time Monitoring":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="dashboard-header">

            <div>

                <div class="dashboard-title">
                    Real-Time Fraud Detection
                </div>

                <div class="dashboard-subtitle">
                    Live transaction monitoring and fraud
                    classification from the Kafka transaction
                    stream.
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # READ KAFKA ONLY HERE
    # --------------------------------------------------------

    new_transactions = (
        read_kafka_transactions()
    )


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


    # --------------------------------------------------------
    # REAL-TIME TRANSACTIONS
    # --------------------------------------------------------

    transactions = list(
        st.session_state.realtime_transactions
    )


    received_count = len(
        transactions
    )


    fraud_count = sum(

        1

        for x in transactions

        if x.get(
            "prediction"
        ) == "FRAUD"

    )


    legitimate_count = sum(

        1

        for x in transactions

        if x.get(
            "prediction"
        ) == "LEGITIMATE"

    )


    fraud_rate = (

        fraud_count
        / received_count
        * 100

        if received_count

        else 0

    )


    # ========================================================
    # REAL-TIME KPI CARDS
    # ========================================================

    k1, k2, k3, k4, k5 = st.columns(5)


    # --------------------------------------------------------
    # RECEIVED
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FRAUD
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # LEGITIMATE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FRAUD RATE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # LATEST RISK
    # --------------------------------------------------------

    with k5:

        last_probability = 0.0


        if (
            st.session_state.last_transaction
        ):

            last_probability = (

                st.session_state
                .last_transaction
                .get(
                    "fraud_probability",
                    0
                )

            )


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
                    Fraud probability of latest event
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # RISK + LATEST TRANSACTION
    # ========================================================

    r1, r2 = st.columns([1, 1])


    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    with r1:

        st.markdown(
            "### Real-Time Risk Distribution"
        )


        if received_count > 0:

            st.plotly_chart(

                create_donut_chart(

                    fraud_count,

                    legitimate_count,

                    ""

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


    # --------------------------------------------------------
    # LATEST TRANSACTION
    # --------------------------------------------------------

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
                            Immediate risk-based review
                            recommended.
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
                    "Transaction received but could not "
                    "be classified."
                )


            # ------------------------------------------------
            # TRANSACTION DETAILS
            # ------------------------------------------------

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


    # ========================================================
    # RECENT TRANSACTIONS
    # ========================================================

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


        table_df = pd.DataFrame(
            table_data
        )


        st.dataframe(

            table_df,

            width="stretch",

            height=420,

            hide_index=True

        )


    else:

        st.info(
            "No Kafka transactions have been received yet."
        )


    # ========================================================
    # REFRESH
    # ========================================================

    st.caption(
        "Kafka is polled only while this page is active. "
        "Switching to another dashboard page does not "
        "consume Kafka messages."
    )


    if st.button(
        "🔄 Refresh Real-Time Data"
    ):

        st.rerun()


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="dashboard-header">

            <div>

                <div class="dashboard-title">
                    Explainability & Risk Interpretation
                </div>

                <div class="dashboard-subtitle">
                    Supporting transparent interpretation of
                    fraud predictions generated by the
                    Random Forest model.
                </div>

            </div>

            <div class="live-badge">
                ● XAI READY
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

        # ----------------------------------------------------
        # GLOBAL FEATURE IMPORTANCE
        # ----------------------------------------------------

        if hasattr(
            model,
            "feature_importances_"
        ):

            feature_names = (
                get_model_features()
            )


            importance = (
                model.feature_importances_
            )


            importance_df = pd.DataFrame(

                {

                    "Feature":
                        feature_names,

                    "Importance":
                        importance

                }

            ).sort_values(

                "Importance",

                ascending=False

            )


            st.markdown(
                "### Global Feature Importance"
            )


            fig = px.bar(

                importance_df.head(15),

                x="Importance",

                y="Feature",

                orientation="h"

            )


            fig.update_layout(

                height=500,

                yaxis=dict(

                    categoryorder=
                        "total ascending"

                )

            )


            st.plotly_chart(

                fig,

                width="stretch",

                config={
                    "displayModeBar": False
                }

            )


            st.markdown(
                """
                **Interpretation**

                Feature importance provides an overview of the
                variables that contribute most strongly to the
                Random Forest's classification behaviour.

                For individual transaction-level explanations,
                SHAP or LIME can be applied to a selected
                transaction.
                """
            )


        else:

            st.info(
                "Feature importance is not available "
                "for this loaded model."
            )


# ============================================================
# SYSTEM MONITORING
# ============================================================

elif page == "System Monitoring":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="dashboard-header">

            <div>

                <div class="dashboard-title">
                    System Monitoring
                </div>

                <div class="dashboard-subtitle">
                    Operational status of the fraud-detection
                    decision-support pipeline.
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    kafka_connected = (

        st.session_state.kafka_status
        == "Connected"

    )


    system_items = [

        (
            "Random Forest Model",
            model is not None
        ),

        (
            "Feature Scaler",
            scaler is not None
        ),

        (
            "Historical Dataset",
            historical_df is not None
        ),

        (
            "Kafka Event Streaming",
            kafka_connected
        ),

        (
            "Real-Time Classification",
            model is not None
        ),

        (
            "Decision Support Dashboard",
            True
        )

    ]


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # PIPELINE STATUS
    # --------------------------------------------------------

    with col1:

        st.markdown(
            "### Pipeline Status"
        )


        for name, active in system_items:

            status = (

                "● ACTIVE"

                if active

                else "● UNAVAILABLE"

            )


            css_class = (

                "status-good"

                if active

                else "status-danger"

            )


            st.markdown(
                f"""
                <div class="dashboard-panel"
                     style="margin-bottom:10px;">

                    <div class="status-row">

                        <span>
                            {name}
                        </span>

                        <span class="{css_class}">
                            {status}
                        </span>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # RUNTIME INFORMATION
    # --------------------------------------------------------

    with col2:

        st.markdown(
            "### Runtime Information"
        )


        st.markdown(
            f"""
            <div class="dashboard-panel">

                <div class="status-row">

                    <span>
                        Kafka Server
                    </span>

                    <span>
                        {KAFKA_SERVER}
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
                        Kafka Status
                    </span>

                    <span>
                        {st.session_state.kafka_status}
                    </span>

                </div>

                <div class="status-row">

                    <span>
                        Session Transactions
                    </span>

                    <span>
                        {len(
                            st.session_state
                            .realtime_transactions
                        )}
                    </span>

                </div>

                <div class="status-row">

                    <span>
                        Maximum Stored Transactions
                    </span>

                    <span>
                        {MAX_REALTIME_TRANSACTIONS}
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
                        Feature Importance / SHAP / LIME
                    </span>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # ARCHITECTURE STATUS
    # ========================================================

    st.markdown(
        "### Architecture Status"
    )


    st.markdown(
        """
        <div class="dashboard-panel">

            <div class="insight-text"
                 style="
                    font-size:13px;
                    line-height:1.8;
                 ">

                The dashboard separates the historical
                analytical layer from the real-time
                operational layer.

                <br><br>

                <b>Historical Layer</b>

                <br>

                Historical transactions → preprocessing →
                Random Forest training/evaluation →
                model metrics → analytical dashboard.

                <br><br>

                <b>Real-Time Layer</b>

                <br>

                Transaction event → Kafka → dashboard consumer →
                preprocessing → Random Forest prediction →
                fraud probability → operational decision support
                → explainability.

                <br><br>

                Kafka polling is intentionally restricted to the
                Real-Time Monitoring page so that the Executive
                Overview and Historical Analysis pages remain
                independent of the live event stream.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# END OF DASHBOARD
# ============================================================