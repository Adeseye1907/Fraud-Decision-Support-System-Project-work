# ============================================================
# INTERPRETABLE FRAUD DETECTION & OPERATIONAL DECISION SUPPORT
# COMPLETE DASHBOARD
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import plotly.graph_objects as go

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

    .kpi-card {
        background: linear-gradient(145deg, #111116, #18181f);
        border: 1px solid #292933;
        border-radius: 12px;
        padding: 18px;
        min-height: 115px;
    }

    .kpi-label {
        color: #8c8c98;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
        color: #ffffff;
    }

    .kpi-description {
        font-size: 11px;
        color: #777783;
        margin-top: 5px;
    }

    .status-card {
        background: #111116;
        border: 1px solid #292933;
        border-radius: 12px;
        padding: 18px;
    }

    .status-row {
        display: flex;
        justify-content: space-between;
        padding: 9px 0;
        border-bottom: 1px solid #24242c;
        font-size: 13px;
    }

    .status-row:last-child {
        border-bottom: none;
    }

    .status-good {
        color: #62d68b;
        font-weight: 600;
    }

    .insight-card {
        background: #111116;
        border: 1px solid #292933;
        border-radius: 12px;
        padding: 20px;
        min-height: 170px;
    }

    .insight-title {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .insight-text {
        color: #a0a0aa;
        font-size: 13px;
        line-height: 1.6;
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

if "last_drift_status" not in st.session_state:
    st.session_state.last_drift_status = "Not evaluated"

if "last_drift_score" not in st.session_state:
    st.session_state.last_drift_score = 0.0


# ============================================================
# KAFKA CONFIGURATION
# ============================================================

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "fraud-predictions"


# ============================================================
# MODEL FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


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
                json.loads(message.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id="fraud-dashboard-consumer-v5",
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

                transaction_id = transaction.get(
                    "transaction_id"
                )

                existing_ids = {
                    t.get("transaction_id")
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
    # SUMMARY
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

        except Exception:

            summary = None

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    if os.path.exists(
        "historical_predictions.csv"
    ):

        try:

            predictions = pd.read_csv(
                "historical_predictions.csv"
            )

        except Exception:

            predictions = None

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

        except Exception:

            shap_importance = None

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

    # --------------------------------------------------------
    # LOAD RANDOM FOREST
    # --------------------------------------------------------

    model = joblib.load(
        "rf_fraud_model.pkl"
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_explainer = shap.TreeExplainer(
        model
    )

    # --------------------------------------------------------
    # LIME BACKGROUND DATA
    # --------------------------------------------------------

    historical_data = pd.read_csv(
        "historical_test_data.csv"
    )

    background_data = historical_data[
        FEATURE_COLUMNS
    ]

    # --------------------------------------------------------
    # LIME EXPLAINER
    # --------------------------------------------------------

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
# SHAP EXTRACTION FUNCTION
# ============================================================

def extract_fraud_shap(
    shap_values
):

    if isinstance(
        shap_values,
        list
    ):

        return np.asarray(
            shap_values[1]
        )[0]

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:

        return shap_values[
            0,
            :,
            1
        ]

    elif shap_values.ndim == 2:

        return shap_values[
            0
        ]

    else:

        return shap_values


# ============================================================
# PREDICT + EXPLAIN
# ============================================================

def predict_and_explain(
    input_df: pd.DataFrame
):

    if pred_model is None:

        raise RuntimeError(
            "Prediction model is not loaded."
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = pred_model.predict(
        input_df
    )[0]

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probability = pred_model.predict_proba(
        input_df
    )[0][1]

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    result = (
        "FRAUD"
        if prediction == 1
        else "LEGITIMATE"
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_values = pred_explainer.shap_values(
        input_df
    )

    fraud_shap = extract_fraud_shap(
        shap_values
    )

    shap_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
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

    # --------------------------------------------------------
    # LIME
    # --------------------------------------------------------

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
# REAL-TIME TRANSACTION → MODEL INPUT
# ============================================================

def transaction_to_model_input(
    transaction
):

    values = {}

    for feature in FEATURE_COLUMNS:

        value = transaction.get(
            feature,
            0.0
        )

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            value = 0.0

        values[feature] = value

    return pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS
    )


# ============================================================
# REAL-TIME LIME EXPLANATION
# ============================================================

def explain_realtime_lime(
    transaction
):

    if (
        lime_explainer is None
        or pred_model is None
    ):

        return None

    try:

        input_df = transaction_to_model_input(
            transaction
        )

        explanation = (
            lime_explainer.explain_instance(
                input_df.iloc[0].values,
                pred_model.predict_proba,
                num_features=10
            )
        )

        lime_df = pd.DataFrame(
            explanation.as_list(),
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

        lime_df[
            "Direction"
        ] = np.where(
            lime_df[
                "LIME_Weight"
            ] >= 0,
            "Pushes toward Fraud",
            "Pushes toward Legitimate"
        )

        lime_df = lime_df.sort_values(
            "Absolute_LIME",
            ascending=False
        )

        return lime_df

    except Exception as e:

        st.warning(
            f"Real-time LIME explanation unavailable: {e}"
        )

        return None


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    # --------------------------------------------------------
    # EXECUTIVE HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="dashboard-header">

            <div>
                <div class="dashboard-title">
                    Executive Dashboard
                </div>

                <div class="dashboard-subtitle">
                    Fraud Detection & Operational Decision Support System
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

                This dashboard provides a high-level view of fraud
                exposure, model performance and the operational
                decision-support pipeline. Historical indicators
                summarise the evaluated transaction dataset, while
                system readiness reflects the availability of the
                supporting detection components.

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
    # KPI 1 — TRANSACTIONS
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
                    {format_number(HISTORICAL_TRANSACTIONS)}
                </div>

                <div class="kpi-description">
                    Historical transactions analysed
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # KPI 2 — FRAUD CASES
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
                    {format_number(HISTORICAL_FRAUD)}
                </div>

                <div class="kpi-description">
                    Identified fraudulent transactions
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # KPI 3 — FRAUD RATE
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
    # KPI 4 — ROC-AUC
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
    # KPI 5 — PR-AUC
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
                    Fraud-class discrimination
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
    # TRANSACTION RISK DISTRIBUTION
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
    # MODEL PERFORMANCE
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

        performance_fig = create_model_performance_chart()

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
    # MODEL EVALUATION SUMMARY
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
                Current availability of the major components
                supporting the fraud-detection workflow.
            </div>
            """,
            unsafe_allow_html=True
        )

        # IMPORTANT:
        # We deliberately do not create or poll Kafka here.
        # This page only reports readiness.

        model_ready = model is not None
        scaler_ready = scaler is not None
        data_ready = historical_df is not None

        st.markdown(
            f"""
            <div class="dashboard-panel">

                <div class="status-row">
                    <span>
                        Historical Transaction Data
                    </span>

                    <span class="{
                        "status-good"
                        if data_ready
                        else "status-danger"
                    }">
                        ● {
                            "AVAILABLE"
                            if data_ready
                            else "UNAVAILABLE"
                        }
                    </span>
                </div>

                <div class="status-row">
                    <span>
                        Random Forest Model
                    </span>

                    <span class="{
                        "status-good"
                        if model_ready
                        else "status-danger"
                    }">
                        ● {
                            "READY"
                            if model_ready
                            else "UNAVAILABLE"
                        }
                    </span>
                </div>

                <div class="status-row">
                    <span>
                        Feature Scaling
                    </span>

                    <span class="{
                        "status-good"
                        if scaler_ready
                        else "status-warning"
                    }">
                        ● {
                            "READY"
                            if scaler_ready
                            else "NOT LOADED"
                        }
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

                    <span class="{
                        "status-good"
                        if model_ready
                        else "status-danger"
                    }">
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
    # INSIGHT 1 — FRAUD EXPOSURE
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
                    highlighting the highly imbalanced nature of
                    fraudulent activity.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # INSIGHT 2 — DETECTION CAPABILITY
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
    # INSIGHT 3 — OPERATIONAL CONSIDERATION
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
    # EXECUTIVE FOOTNOTE
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
        "Interpretable Fraud Detection & Operational Decision Support"
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
# 1. INTRODUCTION
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
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            Real-Time Intelligence for Digital Banking
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="research-title">
        <b>
        Development of Adaptive and Interpretable Machine Learning
        Framework for Real-Time Fraud Detection and Operational Decision
        Support in Digital Banking
        </b>
        <br><br>
        Developed by <b>Adeseye Samuel Ademola</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Background of the Study</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The rapid growth of digital banking and electronic payment systems
        has increased the volume, velocity and complexity of financial
        transactions. At the same time, this growth has created opportunities
        for increasingly sophisticated fraudulent activities.

        Machine learning provides an opportunity to identify complex patterns
        within transaction data and detect potentially fraudulent behaviour.
        However, predictive performance alone is insufficient for effective
        operational fraud management.

        Operational teams need to understand not only whether a transaction
        has been classified as potentially fraudulent, but also why the
        machine learning model reached that decision.

        This framework therefore combines fraud prediction, explainable AI,
        real-time monitoring, adaptive monitoring and operational decision
        support.
        """
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Research Focus</div>',
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
            "Provides insight into model behaviour and identifies factors contributing to individual predictions."
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
        ),
    ]

    workflow_cols = st.columns(4)

    for i, item in enumerate(workflow):

        with workflow_cols[i % 4]:

            st.markdown(
                f"### {item[0]}"
            )

            st.write(
                f"**{item[1]}**"
            )

            st.caption(
                item[2]
            )

    st.divider()

    st.markdown(
        '<div class="section-title">Model Input</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The trained fraud detection model operates on 30 transaction
        features consisting of **Time, V1-V28 and Amount**.
        """
    )





# ============================================================
# 3. HISTORICAL MODEL ANALYSIS
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
        "Evaluation and interpretation of the trained Random Forest fraud detection model."
    )

    st.divider()

    try:

        with open(
            "historical_summary.json",
            "r"
        ) as f:

            historical_summary_page = json.load(
                f
            )

        historical_predictions_page = pd.read_csv(
            "historical_predictions.csv"
        )

        historical_shap_page = pd.read_csv(
            "historical_global_shap.csv"
        )

        files_loaded = True

    except Exception as e:

        files_loaded = False

        st.error(
            f"Historical analysis files could not be loaded: {e}"
        )

    if files_loaded:

        total_transactions = int(
            historical_summary_page.get(
                "total_transactions",
                0
            )
        )

        actual_fraud_count = int(
            historical_summary_page.get(
                "actual_fraud_count",
                0
            )
        )

        fraud_caught = int(
            historical_summary_page.get(
                "fraud_caught",
                0
            )
        )

        fraud_missed = int(
            historical_summary_page.get(
                "fraud_missed",
                0
            )
        )

        false_alarms = int(
            historical_summary_page.get(
                "false_alarms",
                0
            )
        )

        detection_rate = float(
            historical_summary_page.get(
                "detection_rate",
                0
            )
        )

        roc_auc = float(
            historical_summary_page.get(
                "roc_auc",
                0
            )
        )

        pr_auc = float(
            historical_summary_page.get(
                "pr_auc",
                0
            )
        )

        fraud_precision = float(
            historical_summary_page.get(
                "fraud_precision",
                0
            )
        )

        fraud_recall = float(
            historical_summary_page.get(
                "fraud_recall",
                0
            )
        )

        fraud_f1 = float(
            historical_summary_page.get(
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
                "Fraud F1-Score",
                f"{fraud_f1:.3f}"
            )

        st.divider()

        st.subheader(
            "Global Model Interpretability"
        )

        st.image(
            XAI_ICON,
            width=60
        )

        st.info(
            "Global SHAP identifies the features with the greatest influence on model predictions."
        )

        st.dataframe(
            historical_shap_page.head(10),
            use_container_width=True,
            hide_index=True
        )

        if (
            "Feature" in historical_shap_page.columns
            and "Mean_Abs_SHAP" in historical_shap_page.columns
        ):

            chart_df = (
                historical_shap_page
                .head(10)
                .sort_values(
                    "Mean_Abs_SHAP"
                )
                .set_index(
                    "Feature"
                )
            )

            st.bar_chart(
                chart_df[
                    "Mean_Abs_SHAP"
                ]
            )

        st.divider()

        st.subheader(
            "Prediction Outcome Summary"
        )

        if "prediction" in historical_predictions_page.columns:

            counts = (
                historical_predictions_page[
                    "prediction"
                ].value_counts()
            )

        else:

            counts = pd.Series(
                dtype=int
            )

        p1, p2 = st.columns(2)

        with p1:

            st.metric(
                "Predicted Fraud",
                f"{int(counts.get('FRAUD', 0)):,}"
            )

        with p2:

            st.metric(
                "Predicted Legitimate",
                f"{int(counts.get('LEGITIMATE', 0)):,}"
            )

        st.divider()

        st.subheader(
            "Historical Prediction Results"
        )

        display_columns = [
            "prediction",
            "fraud_probability",
            "actual_label"
        ]

        available_columns = [
            c
            for c in display_columns
            if c in historical_predictions_page.columns
        ]

        st.dataframe(
            historical_predictions_page[
                available_columns
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 4. PREDICT FRAUD
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
        "Predict fraud for an individual transaction or a batch of transactions and examine SHAP and LIME explanations."
    )

    st.divider()

    if model_error:

        st.error(
            f"Model could not be loaded: {model_error}"
        )

    else:

        tab_manual, tab_batch = st.tabs(
            [
                "Manual Entry",
                "Batch CSV Upload"
            ]
        )

        # ----------------------------------------------------
        # MANUAL ENTRY
        # ----------------------------------------------------

        with tab_manual:

            st.subheader(
                "Enter Transaction Details"
            )

            st.caption(
                "V1-V28 are anonymised PCA-transformed features. Zero values can be used for testing."
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

                v_cols = st.columns(4)

                for i in range(1, 29):

                    col = v_cols[
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
                    "Time": input_time
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

                try:

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
                            f"FRAUD DETECTED - Probability: {probability:.2%}"
                        )

                    else:

                        st.success(
                            f"LEGITIMATE - Fraud Probability: {probability:.2%}"
                        )

                    st.subheader(
                        "SHAP Explanation"
                    )

                    shap_display = shap_df.head(
                        10
                    ).copy()

                    shap_display[
                        "Direction"
                    ] = np.where(
                        shap_display[
                            "SHAP_Value"
                        ] >= 0,
                        "Pushes toward Fraud",
                        "Pushes toward Legitimate"
                    )

                    st.dataframe(
                        shap_display[
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

                    st.subheader(
                        "LIME Explanation"
                    )

                    lime_display = lime_df.copy()

                    lime_display[
                        "Direction"
                    ] = np.where(
                        lime_display[
                            "LIME_Weight"
                        ] >= 0,
                        "Pushes toward Fraud",
                        "Pushes toward Legitimate"
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

                except Exception as e:

                    st.error(
                        f"Prediction/explanation failed: {e}"
                    )

        # ----------------------------------------------------
        # BATCH
        # ----------------------------------------------------

        with tab_batch:

            st.subheader(
                "Upload a CSV of Transactions"
            )

            st.caption(
                "Required columns: Time, V1-V28 and Amount."
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

                missing_cols = [
                    c
                    for c in FEATURE_COLUMNS
                    if c not in batch_df.columns
                ]

                if missing_cols:

                    st.error(
                        f"Missing required columns: {missing_cols}"
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

                    results_df = batch_df.copy()

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

                    total_batch = len(
                        results_df
                    )

                    fraud_batch = int(
                        (
                            predictions == 1
                        ).sum()
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "Transactions",
                            total_batch
                        )

                    with c2:

                        st.metric(
                            "Flagged Fraud",
                            fraud_batch
                        )

                    with c3:

                        rate = (
                            fraud_batch
                            / total_batch
                            * 100
                            if total_batch
                            else 0
                        )

                        st.metric(
                            "Fraud Rate",
                            f"{rate:.2f}%"
                        )

                    st.subheader(
                        "Flagged Transactions"
                    )

                    fraud_results = (
                        results_df[
                            results_df[
                                "prediction"
                            ] == "FRAUD"
                        ]
                        .sort_values(
                            "fraud_probability",
                            ascending=False
                        )
                    )

                    if fraud_results.empty:

                        st.info(
                            "No transactions were flagged as fraud."
                        )

                    else:

                        st.dataframe(
                            fraud_results,
                            use_container_width=True,
                            hide_index=True
                        )

                    st.subheader(
                        "Full Results"
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
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
# 5. REAL-TIME FRAUD DETECTION
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
        "Live transaction monitoring and fraud classification from the Kafka transaction stream."
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
                f"FRAUD DETECTED - Probability: {probability:.2%}"
            )

        elif prediction == "LEGITIMATE":

            st.success(
                f"LEGITIMATE TRANSACTION - Fraud Probability: {probability:.2%}"
            )

        else:

            st.warning(
                "Transaction classification unavailable."
            )

        st.divider()

        st.subheader(
            "Latest Transaction"
        )

        st.json(
            latest
        )

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

        st.subheader(
            "Real-Time LIME Explanation"
        )

        lime_realtime = (
            explain_realtime_lime(
                latest
            )
        )

        if lime_realtime is None:

            st.info(
                "LIME explanation is unavailable for this transaction. The Kafka message must contain the model features Time, V1-V28 and Amount."
            )

        else:

            st.dataframe(
                lime_realtime[
                    [
                        "Feature",
                        "LIME_Weight",
                        "Direction"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 6. TRANSACTION EXPLORER
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
        "Investigate individual transactions, predictions and risk information."
    )

    st.divider()

    query = st.text_input(
        "Transaction ID",
        placeholder="Enter transaction ID..."
    )

    transactions = (
        st.session_state.transactions
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

            st.success(
                f"{len(matches)} transaction(s) found."
            )

            st.dataframe(
                pd.DataFrame(matches),
                use_container_width=True,
                hide_index=True
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
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No transaction data available."
        )


# ============================================================
# 7. TRENDS & MONITORING
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
        "Monitor transaction activity and fraud patterns across the real-time transaction stream."
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
                subset=[
                    "timestamp"
                ]
            )

            if not df.empty:

                chart = (
                    df
                    .set_index(
                        "timestamp"
                    )
                    .resample(
                        "1min"
                    )
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
                "Timestamp information is not available."
            )

    else:

        st.info(
            "Waiting for transaction data..."
        )


# ============================================================
# 8. ADAPTIVE MONITORING
# ============================================================

elif current_page == "Adaptive Monitoring":

    st.image(
        TREND_ICON,
        width=65
    )

    st.title(
        "Adaptive Monitoring & Drift Detection"
    )

    st.write(
        """
        Monitors incoming transaction behaviour and provides an early
        indication of whether the operating environment may have changed.
        """
    )

    st.divider()

    transactions = (
        st.session_state.transactions
    )

    transaction_count = len(
        transactions
    )

    st.metric(
        "Real-Time Transactions Available",
        transaction_count
    )

    st.divider()

    drift_score = 0.0
    drift_reasons = []

    if transaction_count >= 10:

        live_df = pd.DataFrame(
            transactions
        )

        if "prediction" in live_df.columns:

            fraud_rate = (
                (
                    live_df[
                        "prediction"
                    ] == "FRAUD"
                ).mean()
            )

            if fraud_rate > 0.20:

                drift_score += 0.50

                drift_reasons.append(
                    "Elevated fraud prediction rate"
                )

            elif fraud_rate > 0.10:

                drift_score += 0.25

                drift_reasons.append(
                    "Moderately elevated fraud prediction rate"
                )

        if "fraud_probability" in live_df.columns:

            probabilities = pd.to_numeric(
                live_df[
                    "fraud_probability"
                ],
                errors="coerce"
            ).dropna()

            if not probabilities.empty:

                mean_probability = (
                    probabilities.mean()
                )

                if mean_probability > 0.30:

                    drift_score += 0.50

                    drift_reasons.append(
                        "Elevated average fraud probability"
                    )

                elif mean_probability > 0.15:

                    drift_score += 0.25

                    drift_reasons.append(
                        "Moderately elevated average fraud probability"
                    )

    else:

        drift_reasons.append(
            "Insufficient real-time transactions for reliable monitoring"
        )

    drift_score = min(
        drift_score,
        1.0
    )

    st.session_state.last_drift_score = (
        drift_score
    )

    if transaction_count < 10:

        drift_status = "INSUFFICIENT DATA"

    elif drift_score >= 0.50:

        drift_status = "ADAPTATION REQUIRED"

    elif drift_score >= 0.25:

        drift_status = "MONITOR CLOSELY"

    else:

        drift_status = "STABLE"

    st.session_state.last_drift_status = (
        drift_status
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Drift Score",
            f"{drift_score:.2f}"
        )

    with c2:

        st.metric(
            "Monitoring Status",
            drift_status
        )

    st.divider()

    st.subheader(
        "Monitoring Assessment"
    )

    for reason in drift_reasons:

        st.write(
            f"• {reason}"
        )

    if drift_status == "ADAPTATION REQUIRED":

        st.error(
            """
            Adaptation Required: the monitoring indicators suggest that
            transaction behaviour may have changed sufficiently to warrant
            model review and possible retraining.
            """
        )

    elif drift_status == "MONITOR CLOSELY":

        st.warning(
            """
            Monitor Closely: early indications of behavioural change have
            been detected.
            """
        )

    elif drift_status == "STABLE":

        st.success(
            "Current monitoring indicators do not indicate significant drift."
        )

    else:

        st.info(
            "Collect additional transactions before making an adaptation decision."
        )


# ============================================================
# 9. MODEL ADAPTATION
# ============================================================

elif current_page == "Model Adaptation":

    st.image(
        MODEL_ICON,
        width=65
    )

    st.title(
        "Model Adaptation / Update Mechanism"
    )

    st.write(
        """
        This component translates monitoring evidence into an operational
        model-review and adaptation decision.
        """
    )

    st.divider()

    drift_score = (
        st.session_state.last_drift_score
    )

    drift_status = (
        st.session_state.last_drift_status
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Current Drift Score",
            f"{drift_score:.2f}"
        )

    with c2:

        st.metric(
            "Current Status",
            drift_status
        )

    st.divider()

    st.subheader(
        "Adaptation Decision"
    )

    if drift_status == "ADAPTATION REQUIRED":

        st.error(
            "Model adaptation is recommended."
        )

        st.write(
            """
            Recommended process:

            1. Review recent transaction behaviour.
            2. Validate operational feedback.
            3. Assemble an updated training dataset.
            4. Retrain candidate model.
            5. Compare candidate model against the current model.
            6. Validate performance and explainability.
            7. Deploy only if validation criteria are satisfied.
            """
        )

        if st.button(
            "Record Adaptation Review",
            use_container_width=True
        ):

            st.session_state.adaptation_history.append(
                {
                    "timestamp":
                        pd.Timestamp.now().isoformat(),

                    "drift_score":
                        drift_score,

                    "status":
                        "Adaptation review initiated"
                }
            )

            st.success(
                "Adaptation review recorded."
            )

    elif drift_status == "MONITOR CLOSELY":

        st.warning(
            "Continue monitoring before initiating model adaptation."
        )

    else:

        st.success(
            "No immediate model update is indicated."
        )

    st.divider()

    st.subheader(
        "Adaptation History"
    )

    if st.session_state.adaptation_history:

        st.dataframe(
            pd.DataFrame(
                st.session_state.adaptation_history
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No adaptation reviews have been recorded."
        )


# ============================================================
# 10. OPERATIONAL FEEDBACK
# ============================================================

elif current_page == "Operational Feedback":

    st.image(
        INFO_ICON,
        width=65
    )

    st.title(
        "Operational Feedback Mechanism"
    )

    st.write(
        """
        Capture human review outcomes so that operational knowledge can
        become part of the future model-learning process.
        """
    )

    st.divider()

    transactions = (
        st.session_state.transactions
    )

    if not transactions:

        st.info(
            "No real-time transactions are currently available for review."
        )

    else:

        transaction_options = [
            str(
                t.get(
                    "transaction_id",
                    f"Transaction {i + 1}"
                )
            )
            for i, t in enumerate(
                transactions
            )
        ]

        selected_id = st.selectbox(
            "Select Transaction",
            transaction_options
        )

        selected_transaction = None

        for t in transactions:

            if str(
                t.get(
                    "transaction_id",
                    ""
                )
            ) == selected_id:

                selected_transaction = t

                break

        if selected_transaction is not None:

            st.subheader(
                "Transaction Under Review"
            )

            st.json(
                selected_transaction
            )

            with st.form(
                "operational_feedback_form"
            ):

                reviewer_decision = st.selectbox(
                    "Operational Review Outcome",
                    [
                        "Confirmed Fraud",
                        "Confirmed Legitimate",
                        "Requires Further Investigation"
                    ]
                )

                reviewer_comment = st.text_area(
                    "Reviewer Comment"
                )

                submit_feedback = st.form_submit_button(
                    "Submit Feedback",
                    use_container_width=True
                )

            if submit_feedback:

                record = {
                    "timestamp":
                        pd.Timestamp.now().isoformat(),

                    "transaction_id":
                        selected_id,

                    "model_prediction":
                        selected_transaction.get(
                            "prediction",
                            "UNKNOWN"
                        ),

                    "fraud_probability":
                        selected_transaction.get(
                            "fraud_probability",
                            0
                        ),

                    "reviewer_decision":
                        reviewer_decision,

                    "reviewer_comment":
                        reviewer_comment,
                }

                st.session_state.feedback_records.append(
                    record
                )

                st.success(
                    "Operational feedback recorded successfully."
                )

    st.divider()

    st.subheader(
        "Recorded Feedback"
    )

    if st.session_state.feedback_records:

        st.dataframe(
            pd.DataFrame(
                st.session_state.feedback_records
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No operational feedback has been recorded."
        )


# ============================================================
# 11. FEEDBACK → MODEL LEARNING PIPELINE
# ============================================================

elif current_page == "Feedback Learning Pipeline":

    st.image(
        MODEL_ICON,
        width=65
    )

    st.title(
        "Feedback → Model Learning Pipeline"
    )

    st.write(
        """
        Operational feedback provides labelled evidence that can support
        future model evaluation, retraining and adaptation.
        """
    )

    st.divider()

    feedback_count = len(
        st.session_state.feedback_records
    )

    st.metric(
        "Feedback Records Available",
        feedback_count
    )

    st.divider()

    pipeline = pd.DataFrame(
        [
            {
                "Stage": "1",
                "Process": "Operational Review",
                "Purpose": "Human assessment of suspicious transactions",
                "Status": "Available",
            },
            {
                "Stage": "2",
                "Process": "Feedback Capture",
                "Purpose": "Store review outcome and comments",
                "Status": "Available",
            },
            {
                "Stage": "3",
                "Process": "Feedback Validation",
                "Purpose": "Check quality and consistency of feedback",
                "Status": "Planned",
            },
            {
                "Stage": "4",
                "Process": "Training Dataset Update",
                "Purpose": "Add validated observations to future training data",
                "Status": "Planned",
            },
            {
                "Stage": "5",
                "Process": "Model Retraining",
                "Purpose": "Train candidate updated model",
                "Status": "Planned",
            },
            {
                "Stage": "6",
                "Process": "Model Validation",
                "Purpose": "Compare candidate against current model",
                "Status": "Planned",
            },
            {
                "Stage": "7",
                "Process": "Controlled Deployment",
                "Purpose": "Deploy only validated model updates",
                "Status": "Planned",
            },
        ]
    )

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    if feedback_count > 0:

        feedback_df = pd.DataFrame(
            st.session_state.feedback_records
        )

        st.subheader(
            "Feedback Available for Future Learning"
        )

        st.dataframe(
            feedback_df,
            use_container_width=True,
            hide_index=True
        )

        csv_feedback = (
            feedback_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Feedback Dataset",
            data=csv_feedback,
            file_name="operational_feedback.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.info(
            "Operational feedback must be collected before feedback-driven learning can begin."
        )


# ============================================================
# 12. COST / VALUE FRAMING
# ============================================================

elif current_page == "Cost / Value Framing":

    st.image(
        VALUE_PROTECTED_ICON,
        width=65
    )

    st.title(
        "Cost / Value Framing"
    )

    st.write(
        """
        Fraud detection decisions involve trade-offs between fraud losses,
        false alarms and operational investigation costs.
        """
    )

    st.divider()

    st.subheader(
        "Operational Cost Assumptions"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        false_negative_cost = st.number_input(
            "Cost of Missed Fraud",
            min_value=0.0,
            value=100000.0,
            step=10000.0
        )

    with c2:

        false_positive_cost = st.number_input(
            "Cost of False Alarm",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

    with c3:

        investigation_cost = st.number_input(
            "Investigation Cost",
            min_value=0.0,
            value=500.0,
            step=100.0
        )

    st.divider()

    missed_fraud = 0
    false_alarms = 0

    if historical_summary:

        missed_fraud = int(
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

    estimated_missed_fraud_cost = (
        missed_fraud
        * false_negative_cost
    )

    estimated_false_alarm_cost = (
        false_alarms
        * (
            false_positive_cost
            + investigation_cost
        )
    )

    total_operational_cost = (
        estimated_missed_fraud_cost
        + estimated_false_alarm_cost
    )

    v1, v2, v3 = st.columns(3)

    with v1:

        st.metric(
            "Estimated Missed-Fraud Cost",
            f"₦{estimated_missed_fraud_cost:,.2f}"
        )

    with v2:

        st.metric(
            "Estimated False-Alarm Cost",
            f"₦{estimated_false_alarm_cost:,.2f}"
        )

    with v3:

        st.metric(
            "Estimated Operational Cost",
            f"₦{total_operational_cost:,.2f}"
        )

    st.divider()

    st.info(
        """
        These values are illustrative decision-support assumptions.
        They should be replaced with institution-specific estimates when
        operational cost data become available.
        """
    )


# ============================================================
# 13. THRESHOLD TUNER
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
        """
        Explore how the fraud probability threshold affects operational
        classification decisions.
        """
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

    if historical_predictions is not None:

        if "fraud_probability" in historical_predictions.columns:

            temp = historical_predictions.copy()

            temp[
                "threshold_prediction"
            ] = np.where(
                temp[
                    "fraud_probability"
                ] >= threshold,
                "FRAUD",
                "LEGITIMATE"
            )

            threshold_fraud_count = int(
                (
                    temp[
                        "threshold_prediction"
                    ] == "FRAUD"
                ).sum()
            )

            st.metric(
                "Transactions Flagged at Threshold",
                threshold_fraud_count
            )

    st.info(
        """
        A lower threshold increases sensitivity but may increase false
        positives. A higher threshold may reduce false alarms but can
        increase missed fraud.
        """
    )


# ============================================================
# 14. AI DECISION ASSISTANT
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
        """
        An interface for assisting operational users in interpreting fraud
        predictions, explanations, monitoring indicators and feedback.
        """
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

            transactions = (
                st.session_state.transactions
            )

            feedback_count = len(
                st.session_state.feedback_records
            )

            st.write(
                f"""
                Current DSS status:

                • Real-time transactions available: {len(transactions)}
                • Adaptive monitoring status: {st.session_state.last_drift_status}
                • Drift score: {st.session_state.last_drift_score:.2f}
                • Operational feedback records: {feedback_count}

                The assistant interface is connected to the DSS monitoring
                context. A production deployment can connect this layer to
                a dedicated language model and approved operational
                knowledge base.
                """
            )


# ============================================================
# 15. END-TO-END DSS WORKFLOW
# ============================================================

elif current_page == "End-to-End DSS Workflow":

    st.image(
        OVERVIEW_ICON,
        width=65
    )

    st.title(
        "End-to-End DSS Workflow"
    )

    st.write(
        """
        Integrated view of the adaptive and interpretable fraud detection
        and operational decision-support architecture.
        """
    )

    st.divider()

    workflow_df = pd.DataFrame(
        [
            {
                "Stage": "1. Historical Data",
                "Component": "Historical Test Dataset",
                "Function": "Supports model development and evaluation",
                "Status": "Operational",
            },
            {
                "Stage": "2. Fraud Detection",
                "Component": "Random Forest",
                "Function": "Classifies transaction fraud risk",
                "Status": "Operational",
            },
            {
                "Stage": "3. Real-Time Ingestion",
                "Component": "Apache Kafka",
                "Function": "Receives transaction events",
                "Status": "Operational",
            },
            {
                "Stage": "4. Explainability",
                "Component": "SHAP + LIME",
                "Function": "Provides global and local explanations",
                "Status": "Operational",
            },
            {
                "Stage": "5. Adaptive Monitoring",
                "Component": "Drift Detection",
                "Function": "Monitors changes in transaction behaviour",
                "Status": "Operational",
            },
            {
                "Stage": "6. Model Adaptation",
                "Component": "Adaptation Mechanism",
                "Function": "Determines when model review may be required",
                "Status": "Operational",
            },
            {
                "Stage": "7. Operational Feedback",
                "Component": "Human Review",
                "Function": "Captures investigator decisions",
                "Status": "Operational",
            },
            {
                "Stage": "8. Feedback Learning",
                "Component": "Learning Pipeline",
                "Function": "Uses validated feedback for future improvement",
                "Status": "Framework Ready",
            },
            {
                "Stage": "9. Cost / Value",
                "Component": "Decision Support",
                "Function": "Frames fraud risk and operational costs",
                "Status": "Operational",
            },
            {
                "Stage": "10. DSS Integration",
                "Component": "Streamlit Dashboard",
                "Function": "Combines prediction, explanation, monitoring and adaptation",
                "Status": "Operational",
            },
        ]
    )

    st.dataframe(
        workflow_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Current System Status"
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        st.metric(
            "Kafka",
            "Online"
            if consumer is not None
            else "Offline"
        )

    with s2:

        st.metric(
            "Model",
            "Loaded"
            if pred_model is not None
            else "Unavailable"
        )

    with s3:

        st.metric(
            "LIME",
            "Loaded"
            if lime_explainer is not None
            else "Unavailable"
        )

    with s4:

        st.metric(
            "Adaptive Status",
            st.session_state.last_drift_status
        )

    st.divider()

    st.subheader(
        "Adaptive Decision Chain"
    )

    st.write(
        """
        **Real-Time Transactions → Kafka → Fraud Prediction → SHAP/LIME
        Explanation → Adaptive Monitoring → Drift Assessment → Model Review
        → Operational Feedback → Future Model Learning → Cost/Value Assessment
        → Operational Decision Support**
        """
    )


# ============================================================
# 16. ABOUT THE FRAMEWORK
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
        ### Development of Adaptive and Interpretable Machine Learning
        Framework for Real-Time Fraud Detection and Operational Decision
        Support in Digital Banking

        **Developed by: Adeseye Samuel Ademola**
        """
    )

    st.divider()

    st.write(
        """
        The framework integrates machine learning, real-time transaction
        streaming, Explainable Artificial Intelligence, adaptive monitoring,
        operational feedback and decision-support capabilities.
        """
    )

    st.subheader(
        "Core Technologies"
    )

    technologies = pd.DataFrame(
        [
            {
                "Technology": "Python",
                "Purpose": "Application and ML development",
            },
            {
                "Technology": "Pandas / NumPy",
                "Purpose": "Data processing",
            },
            {
                "Technology": "Scikit-learn",
                "Purpose": "Machine learning",
            },
            {
                "Technology": "Random Forest",
                "Purpose": "Fraud classification",
            },
            {
                "Technology": "Apache Kafka",
                "Purpose": "Real-time transaction streaming",
            },
            {
                "Technology": "SHAP",
                "Purpose": "Model explainability",
            },
            {
                "Technology": "LIME",
                "Purpose": "Local interpretable explanations",
            },
            {
                "Technology": "Streamlit",
                "Purpose": "Decision-support interface",
            },
        ]
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
        **Global interpretability:** SHAP global feature importance is used
        to understand the behaviour of the model across historical data.

        **Local interpretability:** SHAP and LIME are used to explain why
        an individual transaction received a particular prediction.
        """
    )

    st.divider()

    st.subheader(
        "Adaptive Framework"
    )

    adaptive_df = pd.DataFrame(
        [
            {
                "Stage": "Transaction Monitoring",
                "Mechanism": "Kafka real-time stream",
                "Purpose": "Continuous transaction observation",
            },
            {
                "Stage": "Drift Detection",
                "Mechanism": "Adaptive monitoring",
                "Purpose": "Identify behavioural change",
            },
            {
                "Stage": "Model Adaptation",
                "Mechanism": "Adaptation trigger",
                "Purpose": "Identify when model review may be required",
            },
            {
                "Stage": "Operational Feedback",
                "Mechanism": "Human review",
                "Purpose": "Capture transaction review outcomes",
            },
            {
                "Stage": "Feedback Learning",
                "Mechanism": "Learning pipeline",
                "Purpose": "Support future model improvement",
            },
            {
                "Stage": "Cost / Value",
                "Mechanism": "Decision support",
                "Purpose": "Balance fraud risk and operational cost",
            },
        ]
    )

    st.dataframe(
        adaptive_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Operational Decision Support"
    )

    st.write(
        """
        The DSS does not replace human decision-making. Instead, it provides
        operational users with predictive, explanatory, monitoring and
        adaptive information that can support transaction investigation,
        prioritisation and fraud-response decisions.
        """
    )

    st.divider()

    st.caption(
        "Interpretable Fraud Detection & Operational Decision Support System"
    )

    st.caption(
        "Developed by Adeseye Samuel Ademola"
    )