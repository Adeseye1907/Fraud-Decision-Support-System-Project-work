# ============================================================
# INTERPRETABLE FRAUD DETECTION & OPERATIONAL DECISION SUPPORT
# COMPLETE DASHBOARD
# ============================================================

import os
import json
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
# ADAPTATION MONITORING SETTINGS
# ============================================================

# Minimum number of recent Kafka transactions required
# before adaptation status is calculated.

MIN_DRIFT_TRANSACTIONS = 30

# Drift thresholds.
#
# STABLE:
#       overall drift < 0.10
#
# MONITOR:
#       0.10 <= overall drift < 0.20
#
# ADAPTATION REQUIRED:
#       overall drift >= 0.20

DRIFT_STABLE_THRESHOLD = 0.10
DRIFT_ADAPTATION_THRESHOLD = 0.20

# Minimum number of features showing significant drift
# before the system strongly recommends adaptation.

MIN_DRIFTED_FEATURES_FOR_ADAPTATION = 3


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
# LOAD HISTORICAL TEST DATA
#
# Used for:
# 1. LIME background
# 2. Adaptive drift reference distribution
# ============================================================

@st.cache_data
def load_historical_test_data():

    try:

        data = pd.read_csv(
            "historical_test_data.csv"
        )

        missing = [
            col
            for col in FEATURE_COLUMNS
            if col not in data.columns
        ]

        if missing:

            return None, (
                "historical_test_data.csv is missing "
                f"required features: {missing}"
            )

        data = data[
            FEATURE_COLUMNS
        ].copy()

        return data, None

    except Exception as e:

        return None, str(e)


(
    historical_test_data,
    historical_data_error
) = load_historical_test_data()


# ============================================================
# LOAD MODEL + SHAP + LIME
# ============================================================

@st.cache_resource
def load_prediction_model():

    # --------------------------------------------------------
    # LOAD RANDOM FOREST MODEL
    # --------------------------------------------------------

    model = joblib.load(
        "rf_fraud_model.pkl"
    )

    # --------------------------------------------------------
    # CREATE SHAP EXPLAINER
    # --------------------------------------------------------

    shap_explainer = shap.TreeExplainer(
        model
    )

    # --------------------------------------------------------
    # CREATE LIME EXPLAINER
    # --------------------------------------------------------

    if historical_test_data is None:

        raise ValueError(
            "historical_test_data.csv could not be loaded. "
            "LIME requires this dataset as its background."
        )

    background_data = historical_test_data[
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
# INITIALISE MODEL VARIABLES
# ============================================================

pred_model = None
pred_explainer = None
lime_explainer = None
model_error = None


# ============================================================
# LOAD MODEL
# ============================================================

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
#
# This function provides:
# - Prediction
# - Fraud probability
# - SHAP explanation
# - LIME explanation
# ============================================================

def predict_and_explain(
    input_df: pd.DataFrame
):

    if pred_model is None:

        raise ValueError(
            "Fraud detection model is not loaded."
        )

    # --------------------------------------------------------
    # Ensure correct feature order
    # --------------------------------------------------------

    input_df = input_df[
        FEATURE_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = pred_model.predict(
        input_df
    )[0]

    # --------------------------------------------------------
    # FRAUD PROBABILITY
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

    # ========================================================
    # SHAP
    # ========================================================

    shap_values = pred_explainer.shap_values(
        input_df
    )

    # --------------------------------------------------------
    # Handle different SHAP versions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SHAP DATAFRAME
    # --------------------------------------------------------

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
# ADAPTIVE MODEL MONITORING
# ============================================================

def calculate_feature_drift(
    historical_df,
    recent_df
):

    results = []

    for feature in FEATURE_COLUMNS:

        historical_values = pd.to_numeric(
            historical_df[feature],
            errors="coerce"
        ).dropna()

        recent_values = pd.to_numeric(
            recent_df[feature],
            errors="coerce"
        ).dropna()

        if len(
            historical_values
        ) == 0 or len(
            recent_values
        ) == 0:

            continue

        historical_mean = (
            historical_values.mean()
        )

        recent_mean = (
            recent_values.mean()
        )

        historical_std = (
            historical_values.std()
        )

        # ----------------------------------------------------
        # Prevent division by zero
        # ----------------------------------------------------

        if (
            pd.isna(historical_std)
            or historical_std == 0
        ):

            historical_std = 1e-9

        # ----------------------------------------------------
        # Standardised mean difference
        #
        # This provides a simple interpretable measure
        # of distributional change.
        # ----------------------------------------------------

        drift_score = abs(
            recent_mean - historical_mean
        ) / (
            abs(historical_std)
            + 1e-9
        )

        results.append(
            {
                "Feature": feature,

                "Historical_Mean":
                    historical_mean,

                "Current_Mean":
                    recent_mean,

                "Historical_Std":
                    historical_std,

                "Drift_Score":
                    drift_score
            }
        )

    if not results:

        return pd.DataFrame()

    drift_df = pd.DataFrame(
        results
    )

    drift_df = drift_df.sort_values(
        "Drift_Score",
        ascending=False
    )

    return drift_df


# ============================================================
# DETERMINE ADAPTATION STATUS
# ============================================================

def determine_adaptation_status(
    drift_df,
    number_of_transactions
):

    # --------------------------------------------------------
    # Not enough data
    # --------------------------------------------------------

    if (
        number_of_transactions
        < MIN_DRIFT_TRANSACTIONS
    ):

        return (
            "INSUFFICIENT DATA",
            "Continue collecting real-time transactions before making an adaptation decision."
        )

    if drift_df.empty:

        return (
            "MONITOR",
            "Drift could not be calculated reliably. Continue monitoring."
        )

    # --------------------------------------------------------
    # Overall drift
    # --------------------------------------------------------

    overall_drift = float(
        drift_df[
            "Drift_Score"
        ].mean()
    )

    # --------------------------------------------------------
    # Number of substantially drifted features
    # --------------------------------------------------------

    drifted_features = int(
        (
            drift_df[
                "Drift_Score"
            ]
            >= DRIFT_ADAPTATION_THRESHOLD
        ).sum()
    )

    # --------------------------------------------------------
    # ADAPTATION REQUIRED
    # --------------------------------------------------------

    if (
        overall_drift
        >= DRIFT_ADAPTATION_THRESHOLD
        or drifted_features
        >= MIN_DRIFTED_FEATURES_FOR_ADAPTATION
    ):

        return (
            "ADAPTATION REQUIRED",

            "Significant changes have been detected "
            "between recent transaction patterns and "
            "the historical reference distribution. "
            "The model should be reassessed and "
            "potentially retrained if the drift persists."
        )

    # --------------------------------------------------------
    # MONITOR
    # --------------------------------------------------------

    elif (
        overall_drift
        >= DRIFT_STABLE_THRESHOLD
    ):

        return (
            "MONITOR",

            "Moderate distributional change has been "
            "detected. Continue monitoring recent "
            "transactions and reassess if the drift increases."
        )

    # --------------------------------------------------------
    # STABLE
    # --------------------------------------------------------

    else:

        return (
            "STABLE",

            "Recent transaction patterns remain "
            "reasonably consistent with the historical "
            "reference distribution."
        )


# ============================================================
# ADAPTATION MONITORING FUNCTION
# ============================================================

def run_adaptation_monitoring():

    if historical_test_data is None:

        return {
            "status": "UNAVAILABLE",
            "message": historical_data_error,
            "drift_df": pd.DataFrame(),
            "overall_drift": 0,
            "transaction_count": 0
        }

    # --------------------------------------------------------
    # Convert Kafka transactions into dataframe
    # --------------------------------------------------------

    transactions = (
        st.session_state.transactions
    )

    if not transactions:

        return {
            "status": "INSUFFICIENT DATA",
            "message": "No real-time transactions are currently available.",
            "drift_df": pd.DataFrame(),
            "overall_drift": 0,
            "transaction_count": 0
        }

    recent_df = pd.DataFrame(
        transactions
    )

    # --------------------------------------------------------
    # Check model feature availability
    # --------------------------------------------------------

    available_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature in recent_df.columns
    ]

    if len(
        available_features
    ) < len(
        FEATURE_COLUMNS
    ):

        missing = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in recent_df.columns
        ]

        return {
            "status": "UNAVAILABLE",
            "message": (
                "The Kafka transactions do not contain "
                f"all model features. Missing: {missing}"
            ),
            "drift_df": pd.DataFrame(),
            "overall_drift": 0,
            "transaction_count": len(recent_df)
        }

    recent_df = recent_df[
        FEATURE_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Convert features to numeric
    # --------------------------------------------------------

    for feature in FEATURE_COLUMNS:

        recent_df[feature] = pd.to_numeric(
            recent_df[feature],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove rows without model features
    # --------------------------------------------------------

    recent_df = recent_df.dropna(
        subset=FEATURE_COLUMNS
    )

    transaction_count = len(
        recent_df
    )

    # --------------------------------------------------------
    # Calculate drift
    # --------------------------------------------------------

    drift_df = calculate_feature_drift(
        historical_test_data,
        recent_df
    )

    # --------------------------------------------------------
    # Overall drift
    # --------------------------------------------------------

    if drift_df.empty:

        overall_drift = 0

    else:

        overall_drift = float(
            drift_df[
                "Drift_Score"
            ].mean()
        )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status, message = (
        determine_adaptation_status(
            drift_df,
            transaction_count
        )
    )

    return {
        "status": status,
        "message": message,
        "drift_df": drift_df,
        "overall_drift": overall_drift,
        "transaction_count": transaction_count
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
            "Adaptive Model Monitoring",
            "Threshold Tuner",
            "AI Decision Assistant",
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
        '<div class="subtitle">Real-Time Intelligence for Digital Banking</div>',
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
        '<div class="section-title">Background of the Study</div>',
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
        artificial intelligence and operational decision support.
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
            "SHAP and LIME provide model and transaction-level explanations."
        ),

        (
            c5,
            MODEL_ICON,
            "05",
            "Adaptive DSS",
            "Drift monitoring identifies changes that may require model reassessment."
        )
    ]

    for col, icon, number, title, caption in workflow:

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
        '<div class="section-title">Model Input</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The trained fraud detection model operates on **30 transaction features consisting of
        Time, Amount and V1-V28**.
        """
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Two Levels of Interpretability</div>',
        unsafe_allow_html=True
    )

    interpretation1, interpretation2 = st.columns(2)

    with interpretation1:

        st.image(
            XAI_ICON,
            width=65
        )

        st.markdown(
            "### Global Interpretability"
        )

        st.write(
            """
            Examines how the model behaves across historical transaction data.

            Examples include:

            - Global SHAP feature importance
            - Feature contribution patterns
            - Model performance
            - ROC-AUC and PR-AUC
            - Precision, recall and F1-score
            """
        )

    with interpretation2:

        st.image(
            TRANSACTION_ICON,
            width=65
        )

        st.markdown(
            "### Local Interpretability"
        )

        st.write(
            """
            Explains why the model produced a particular prediction for an individual transaction.

            Examples include:

            - Transaction fraud probability
            - Individual SHAP contributions
            - LIME local explanation
            - Top risk-contributing features
            """
        )

    st.divider()

    st.markdown(
        '<div class="section-title">Adaptive Monitoring</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The adaptive component monitors the distribution of incoming transactions against the
        historical reference distribution. Where substantial and persistent changes are detected,
        the system can flag **Adaptation Required** and recommend model reassessment.
        """
    )

    st.divider()

    st.info(
        "Use the navigation panel to explore the fraud detection, explainability, real-time streaming and adaptive monitoring components."
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

        st.image(
            TRANSACTION_ICON,
            width=45
        )

        st.metric(
            "Transactions Processed",
            total
        )

    with c2:

        st.image(
            FRAUD_CAUGHT_ICON,
            width=45
        )

        st.metric(
            "Fraud Detected",
            fraud
        )

    with c3:

        st.image(
            OVERVIEW_ICON,
            width=45
        )

        st.metric(
            "Legitimate",
            legitimate
        )

    with c4:

        st.image(
            FRAUD_MISSED_ICON,
            width=45
        )

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
        """
        Evaluation and interpretation of the trained Random Forest
        fraud detection model using the historical test dataset.
        """
    )

    st.divider()

    try:

        historical_summary = pd.read_json(
            "historical_summary.json",
            typ="series"
        )

        historical_predictions = pd.read_csv(
            "historical_predictions.csv"
        )

        historical_shap = pd.read_csv(
            "historical_global_shap.csv"
        )

        historical_files_loaded = True

    except Exception as e:

        historical_files_loaded = False

        st.error(
            f"""
            Historical analysis files could not be loaded.

            Error:
            {e}

            Make sure historical_batch.py has been executed
            successfully.
            """
        )

    if historical_files_loaded:

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

        # ----------------------------------------------------
        # MODEL PERFORMANCE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # HISTORICAL DATA
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # OPERATIONAL PERFORMANCE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # GLOBAL SHAP
        # ----------------------------------------------------

        st.subheader(
            "Global Model Interpretability"
        )

        st.image(
            XAI_ICON,
            width=60
        )

        st.info(
            "Global SHAP analysis identifies features that have the greatest influence on model predictions across the historical test dataset."
        )

        st.markdown(
            "### Top SHAP Features"
        )

        top_shap = historical_shap.head(
            10
        ).copy()

        st.dataframe(
            top_shap,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "### Global Feature Importance"
        )

        shap_chart = (
            historical_shap
            .head(10)
            .sort_values(
                "Mean_Abs_SHAP",
                ascending=True
            )
            .set_index(
                "Feature"
            )
        )

        st.bar_chart(
            shap_chart[
                "Mean_Abs_SHAP"
            ]
        )

        st.divider()

        # ----------------------------------------------------
        # PREDICTION SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Prediction Outcome Summary"
        )

        prediction_counts = (
            historical_predictions[
                "prediction"
            ].value_counts()
        )

        p1, p2 = st.columns(2)

        with p1:

            st.metric(
                "Predicted Fraud",
                f"{int(prediction_counts.get('FRAUD', 0)):,}"
            )

        with p2:

            st.metric(
                "Predicted Legitimate",
                f"{int(prediction_counts.get('LEGITIMATE', 0)):,}"
            )

        st.divider()

        # ----------------------------------------------------
        # HISTORICAL TABLE
        # ----------------------------------------------------

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

        st.divider()

        historical_csv = (
            historical_predictions
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Historical Predictions",
            data=historical_csv,
            file_name="historical_predictions.csv",
            mime="text/csv",
            use_container_width=True
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
        """
        Submit a transaction manually or upload a CSV of transactions
        to obtain fraud predictions and transaction-level explanations
        using SHAP and LIME.
        """
    )

    st.divider()

    if model_error:

        st.error(
            f"Model/SHAP/LIME loading error: {model_error}"
        )

    elif pred_model is None:

        st.error(
            "The prediction model could not be loaded."
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

            st.caption(
                "V1-V28 are anonymised PCA-transformed features. Leave values at 0.0 for testing if necessary."
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

                v_cols_ui = st.columns(
                    4
                )

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

                    # ----------------------------------------
                    # SHAP
                    # ----------------------------------------

                    st.subheader(
                        "SHAP Explanation"
                    )

                    top5 = shap_df.head(
                        5
                    ).copy()

                    top5[
                        "Direction"
                    ] = np.where(
                        top5[
                            "SHAP_Value"
                        ] >= 0,
                        "Pushes toward Fraud",
                        "Pushes toward Legitimate"
                    )

                    st.dataframe(
                        top5[
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

                    # ----------------------------------------
                    # LIME
                    # ----------------------------------------

                    st.subheader(
                        "LIME Local Explanation"
                    )

                    st.info(
                        """
                        LIME explains this individual prediction by approximating
                        the model's local behaviour around the submitted transaction.
                        Positive weights support the fraud class while negative weights
                        support the legitimate class.
                        """
                    )

                    lime_display = lime_df.copy()

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

                except Exception as e:

                    st.error(
                        f"Prediction/explanation failed: {e}"
                    )

        # ====================================================
        # BATCH CSV
        # ====================================================

        with tab_batch:

            st.subheader(
                "Upload a CSV of Transactions"
            )

            st.caption(
                "The CSV must contain: Time, V1-V28 and Amount."
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

                missing_cols = set(
                    FEATURE_COLUMNS
                ) - set(
                    batch_df.columns
                )

                if missing_cols:

                    st.error(
                        f"Uploaded file is missing required columns: {sorted(missing_cols)}"
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

                    total_batch = len(
                        results_df
                    )

                    fraud_batch = int(
                        (
                            predictions == 1
                        ).sum()
                    )

                    bc1, bc2, bc3 = st.columns(
                        3
                    )

                    with bc1:

                        st.metric(
                            "Transactions Uploaded",
                            total_batch
                        )

                    with bc2:

                        st.metric(
                            "Flagged as Fraud",
                            fraud_batch
                        )

                    with bc3:

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

                    st.divider()

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
                            "No transactions in this batch were flagged as fraud."
                        )

                    else:

                        st.dataframe(
                            fraud_results,
                            use_container_width=True
                        )

                    st.divider()

                    st.subheader(
                        "Full Results"
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )

                    csv_download = (
                        results_df
                        .to_csv(
                            index=False
                        )
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
        # REAL-TIME LIME
        # ====================================================

        st.divider()

        st.subheader(
            "Real-Time LIME Explanation"
        )

        # ----------------------------------------------------
        # First use LIME sent through Kafka if available.
        # ----------------------------------------------------

        if latest.get(
            "top_lime_features"
        ):

            lime_live = pd.DataFrame(
                latest[
                    "top_lime_features"
                ]
            )

            st.info(
                "LIME explanation generated for the latest Kafka transaction."
            )

            st.dataframe(
                lime_live,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # Otherwise generate LIME locally from raw features.
        # ----------------------------------------------------

        elif all(
            feature in latest
            for feature in FEATURE_COLUMNS
        ):

            try:

                live_row = {
                    feature: latest[
                        feature
                    ]
                    for feature in FEATURE_COLUMNS
                }

                live_df = pd.DataFrame(
                    [live_row],
                    columns=FEATURE_COLUMNS
                )

                (
                    live_result,
                    live_probability,
                    live_shap_df,
                    live_lime_df
                ) = predict_and_explain(
                    live_df
                )

                st.info(
                    "LIME explanation generated locally from the latest Kafka transaction."
                )

                lime_display = (
                    live_lime_df.copy()
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

            except Exception as e:

                st.warning(
                    f"Unable to generate local LIME explanation: {e}"
                )

        else:

            st.info(
                """
                LIME explanation is not available for this transaction because
                the Kafka message does not contain top_lime_features and does not
                contain all 30 model features required to generate a local explanation.
                """
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
        "Investigate individual transactions, predictions and transaction-level risk information."
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
                pd.DataFrame(
                    matches
                ),
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
# 17. ADAPTIVE MODEL MONITORING
# ============================================================

elif current_page == "Adaptive Model Monitoring":

    st.image(
        MODEL_ICON,
        width=65
    )

    st.title(
        "Adaptive Model Monitoring"
    )

    st.write(
        """
        Monitor changes in incoming transaction patterns against the
        historical reference distribution to determine whether the
        fraud detection model requires reassessment.
        """
    )

    st.divider()

    # ========================================================
    # RUN ADAPTATION MONITORING
    # ========================================================

    adaptation = (
        run_adaptation_monitoring()
    )

    status = adaptation[
        "status"
    ]

    message = adaptation[
        "message"
    ]

    drift_df = adaptation[
        "drift_df"
    ]

    overall_drift = adaptation[
        "overall_drift"
    ]

    transaction_count = adaptation[
        "transaction_count"
    ]

    # ========================================================
    # STATUS
    # ========================================================

    st.subheader(
        "Adaptation Status"
    )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.metric(
            "Recent Transactions",
            transaction_count
        )

    with a2:

        st.metric(
            "Overall Drift Score",
            f"{overall_drift:.3f}"
        )

    with a3:

        if status == "STABLE":

            st.success(
                "STABLE"
            )

        elif status == "MONITOR":

            st.warning(
                "MONITOR"
            )

        elif status == "ADAPTATION REQUIRED":

            st.error(
                "ADAPTATION REQUIRED"
            )

        else:

            st.info(
                status
            )

    st.divider()

    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader(
        "Interpretation"
    )

    if status == "STABLE":

        st.success(
            message
        )

    elif status == "MONITOR":

        st.warning(
            message
        )

    elif status == "ADAPTATION REQUIRED":

        st.error(
            message
        )

    else:

        st.info(
            message
        )

    st.divider()

    # ========================================================
    # DRIFT THRESHOLDS
    # ========================================================

    st.subheader(
        "Adaptation Decision Rules"
    )

    threshold_df = pd.DataFrame(
        {
            "Condition": [
                "Stable",
                "Monitor",
                "Adaptation Required"
            ],

            "Drift Rule": [
                f"Overall drift < {DRIFT_STABLE_THRESHOLD:.2f}",

                (
                    f"{DRIFT_STABLE_THRESHOLD:.2f} "
                    f"≤ Overall drift < "
                    f"{DRIFT_ADAPTATION_THRESHOLD:.2f}"
                ),

                (
                    f"Overall drift ≥ "
                    f"{DRIFT_ADAPTATION_THRESHOLD:.2f} "
                    "OR multiple features show significant drift"
                )
            ],

            "Operational Meaning": [
                "Continue normal monitoring.",
                "Increase monitoring of transaction patterns.",
                "Reassess the model and consider retraining if drift persists."
            ]
        }
    )

    st.dataframe(
        threshold_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # FEATURE DRIFT
    # ========================================================

    st.subheader(
        "Feature-Level Drift"
    )

    if drift_df.empty:

        st.info(
            "Feature-level drift cannot currently be calculated."
        )

    else:

        display_drift = drift_df.copy()

        display_drift[
            "Drift_Status"
        ] = np.where(

            display_drift[
                "Drift_Score"
            ] >= DRIFT_ADAPTATION_THRESHOLD,

            "SIGNIFICANT",

            np.where(

                display_drift[
                    "Drift_Score"
                ] >= DRIFT_STABLE_THRESHOLD,

                "MONITOR",

                "STABLE"
            )
        )

        st.dataframe(
            display_drift[
                [
                    "Feature",
                    "Historical_Mean",
                    "Current_Mean",
                    "Historical_Std",
                    "Drift_Score",
                    "Drift_Status"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # ========================================================
    # TOP DRIFTED FEATURES
    # ========================================================

    st.subheader(
        "Most Drifted Features"
    )

    if not drift_df.empty:

        top_drift = (
            drift_df
            .head(10)
            .set_index(
                "Feature"
            )[
                "Drift_Score"
            ]
        )

        st.bar_chart(
            top_drift
        )

        st.divider()

        st.subheader(
            "Top Features Requiring Attention"
        )

        top_features = drift_df.head(
            5
        ).copy()

        st.dataframe(
            top_features[
                [
                    "Feature",
                    "Historical_Mean",
                    "Current_Mean",
                    "Drift_Score"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No drift features available."
        )

    st.divider()

    # ========================================================
    # OPERATIONAL RECOMMENDATION
    # ========================================================

    st.subheader(
        "Operational Recommendation"
    )

    if status == "ADAPTATION REQUIRED":

        st.error(
            """
            **Model reassessment is recommended.**

            The incoming transaction distribution differs substantially
            from the historical reference distribution. Operational teams
            should investigate the changed transaction patterns and determine
            whether the existing fraud detection model remains appropriate.

            If the drift persists after further monitoring, the model should
            be evaluated for retraining using more recent representative data.
            """
        )

    elif status == "MONITOR":

        st.warning(
            """
            **Continue enhanced monitoring.**

            Some changes have been identified in the incoming transaction
            distribution. Additional transactions should be monitored before
            making a model adaptation decision.
            """
        )

    elif status == "STABLE":

        st.success(
            """
            **No immediate model adaptation is indicated.**

            The current transaction distribution remains reasonably consistent
            with the historical reference distribution.
            """
        )

    else:

        st.info(
            """
            **Continue collecting data.**

            A reliable adaptation decision requires a sufficient number of
            recent real-time transactions.
            """
        )

    st.divider()

    # ========================================================
    # IMPORTANT METHODOLOGICAL NOTE
    # ========================================================

    st.subheader(
        "Adaptive Framework Note"
    )

    st.info(
        """
        The adaptation component is a monitoring and decision-support mechanism.
        It does not automatically retrain the Random Forest model.

        An adaptation flag indicates that the model environment may have changed
        sufficiently to justify model reassessment. Final retraining decisions
        should consider additional evidence such as sustained drift, model
        performance, fraud patterns and availability of representative recent data.
        """
    )


# ============================================================
# 18. THRESHOLD TUNER
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
        "Explore the relationship between fraud probability thresholds and operational classification decisions."
    )

    st.divider()

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
    )

    st.metric(
        "Selected Threshold",
        f"{threshold:.2f}"
    )

    st.divider()

    st.info(
        """
        A lower threshold increases sensitivity to potentially fraudulent
        transactions but may increase false positives.

        A higher threshold may reduce false alarms but can increase the risk
        of missed fraud.

        The final operational threshold should be selected using validation
        data and the relative cost of false positives and false negatives.
        """
    )


# ============================================================
# 19. AI DECISION ASSISTANT
# ============================================================

elif current_page == "AI Decision Assistant":

    st.image(
        AI_ASSISTANT_ICON,
        width=65
    )

    st.title(
        "AI Assistant"
    )

    st.write(
        """
        An intelligent interface for assisting operational users in interpreting
        fraud detection results, transaction risk, model explanations and adaptation alerts.
        """
    )

    st.divider()

    question = st.chat_input(
        "Ask about a transaction, prediction, model explanation or adaptation status..."
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
                The AI Decision Assistant interface is ready.

                The next integration stage can connect the assistant to
                transaction predictions, SHAP explanations, LIME explanations,
                historical model results and adaptation-monitoring outputs.
                """
            )


# ============================================================
# 20. ABOUT THE FRAMEWORK
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
        Explainable Artificial Intelligence, adaptive monitoring and operational
        decision-support capabilities.
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
                "Model explainability",
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
        The framework addresses interpretability at two levels:

        **Global interpretability:** Understanding the behaviour of the model
        across historical transaction data.

        **Local interpretability:** Understanding why the model produced a
        particular prediction for an individual transaction.

        SHAP and LIME are used to provide feature-level explanations.
        """
    )

    st.divider()

    st.subheader(
        "Adaptive Monitoring"
    )

    st.write(
        """
        The framework monitors changes between historical transaction patterns
        and recent real-time transaction data. Significant distributional changes
        can trigger an **Adaptation Required** status, prompting model reassessment.
        """
    )

    st.divider()

    st.subheader(
        "Operational Decision Support"
    )

    st.write(
        """
        The DSS does not replace human decision-making. Instead, it provides
        operational users with predictive, explanatory and monitoring information
        that can support transaction investigation, prioritisation and model governance.
        """
    )

    st.divider()

    st.caption(
        "Interpretable Fraud Detection & Operational Decision Support System"
    )

    st.caption(
        "Developed by Adeseye Samuel Ademola"
    )