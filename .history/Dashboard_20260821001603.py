import streamlit as st
import pandas as pd
import os

# ============================================================
# XAI DSS
# INTERPRETABLE FRAUD DETECTION & OPERATIONAL
# DECISION SUPPORT SYSTEM FOR DIGITAL BANKING
# ============================================================

st.set_page_config(
    page_title="Fraud Detection DSS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ASSET PATH
# ============================================================

ASSET_DIR = "assets"


def asset(filename):
    return os.path.join(ASSET_DIR, filename)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 17px;
    color: #666666;
    margin-top: 0px;
}

.author {
    font-size: 14px;
    color: #777777;
}

.section-title {
    font-size: 24px;
    font-weight: 650;
}

.metric-card {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.image(
        asset("overview.png"),
        width=55
    )

    st.markdown("## Fraud Detection DSS")

    st.caption(
        "Interpretable Machine Learning "
        "for Operational Decision Support"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Introduction",
            "Executive Overview",
            "Real-Time Fraud Detection",
            "Model Performance",
            "XAI & Explainability",
            "Transaction Explorer",
            "Monitoring & Trends",
            "About"
        ]
    )

    st.divider()

    st.caption("SYSTEM STATUS")

    st.success("System Online")

    st.caption("Kafka Stream: Ready")
    st.caption("ML Model: Loaded")
    st.caption("XAI Engine: Loaded")


# ============================================================
# INTRODUCTION
# ============================================================

if page == "Introduction":

    col1, col2 = st.columns([0.15, 0.85])

    with col1:
        st.image(
            asset("overview.png"),
            width=75
        )

    with col2:
        st.markdown(
            '<div class="main-title">'
            'Interpretable Fraud Detection DSS'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">'
            'Development of Adaptive and Interpretable '
            'Machine Learning Framework for Real-Time '
            'Fraud Detection and Operational Decision '
            'Support in Digital Banking'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="author">'
            'Developed by <b>Adeseye Samuel Ademola</b>'
            '</div>',
            unsafe_allow_html=True
        )

    st.divider()

    st.image(
        asset("overview.png"),
        width=120
    )

    st.markdown(
        "## Background of the Study"
    )

    st.markdown("""
Digital banking has transformed financial services by enabling
fast, convenient and increasingly automated financial transactions.
However, the growth of digital transactions has also created
opportunities for conventional and emerging forms of financial fraud.

Traditional fraud detection approaches may struggle to identify
complex and evolving transaction patterns, particularly where
fraudsters continuously adapt their behaviour.

Machine learning provides an opportunity to identify patterns
associated with fraudulent transactions and support automated
risk classification. However, predictive performance alone is
not sufficient for operational banking environments. Operational
teams need to understand **why a transaction has been classified
as suspicious** before taking appropriate action.

This Decision Support System therefore combines machine learning
fraud detection with explainable artificial intelligence (XAI).
The system provides both historical model interpretation and
real-time transaction-level explanations to support operational
decision-making.
""")

    st.divider()

    st.markdown("## Purpose of the Decision Support System")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.image(asset("shield-check.png"), width=55)
        st.subheader("Detect")
        st.write(
            "Identify potentially fraudulent transactions "
            "using a trained machine learning model."
        )

    with p2:
        st.image(asset("brain-circuit.png"), width=55)
        st.subheader("Explain")
        st.write(
            "Interpret model predictions and identify the "
            "transaction features influencing fraud decisions."
        )

    with p3:
        st.image(asset("target.png"), width=55)
        st.subheader("Support Decisions")
        st.write(
            "Provide actionable information to operational "
            "teams for faster and more informed responses."
        )


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

elif page == "Executive Overview":

    col1, col2 = st.columns([0.12, 0.88])

    with col1:
        st.image(asset("overview.png"), width=65)

    with col2:
        st.header("Executive Overview")
        st.caption(
            "High-level fraud risk and operational decision support."
        )

    st.divider()

    # These are currently demonstration values.
    # Later they will be connected to the live transaction stream.

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.image(asset("shield-check.png"), width=40)
        st.metric(
            "Fraud Detected",
            "—"
        )

    with c2:
        st.image(asset("alert-triangle.png"), width=40)
        st.metric(
            "High-Risk Transactions",
            "—"
        )

    with c3:
        st.image(asset("money-shield.png"), width=40)
        st.metric(
            "Value Protected",
            "—"
        )

    with c4:
        st.image(asset("bell-warning.png"), width=40)
        st.metric(
            "False Alerts",
            "—"
        )

    st.divider()

    st.subheader("Fraud Monitoring")

    st.info(
        "Live transaction metrics will appear here once "
        "the Streamlit dashboard is connected to the "
        "Kafka/XAI processing pipeline."
    )


# ============================================================
# REAL-TIME FRAUD DETECTION
# ============================================================

elif page == "Real-Time Fraud Detection":

    col1, col2 = st.columns([0.12, 0.88])

    with col1:
        st.image(asset("live-pulse.png"), width=65)

    with col2:
        st.header("Real-Time Fraud Detection")
        st.caption(
            "Live transaction monitoring through Kafka and ML."
        )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Stream Status",
            "READY"
        )

    with c2:
        st.metric(
            "Transactions Processed",
            "—"
        )

    with c3:
        st.metric(
            "Fraud Alerts",
            "—"
        )

    st.divider()

    st.subheader("Latest Transaction")

    st.info(
        "The latest Kafka transaction and its fraud prediction "
        "will appear here."
    )

    st.divider()

    st.subheader("Operational Decision")

    st.write(
        "The system will provide the operational team with:"
    )

    st.markdown("""
- Fraud/legitimate classification
- Fraud probability
- Transaction identifier
- Risk level
- Key features contributing to the prediction
- Explainable recommendation for investigation
""")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    col1, col2 = st.columns([0.12, 0.88])

    with col1:
        st.image(asset("target.png"), width=65)

    with col2:
        st.header("Model Performance")
        st.caption(
            "Evaluation of the trained fraud detection model."
        )

    st.divider()

    st.subheader("Model Evaluation")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "ROC-AUC",
            "0.961"
        )

    with c2:
        st.metric(
            "PR-AUC",
            "0.813"
        )

    with c3:
        st.metric(
            "Fraud Recall",
            "77%"
        )

    with c4:
        st.metric(
            "Fraud Precision",
            "90%"
        )

    st.divider()

    st.subheader("Classification Performance")

    performance = pd.DataFrame({
        "Metric": [
            "Precision",
            "Recall",
            "F1-Score"
        ],
        "Fraud Class": [
            0.90,
            0.77,
            0.83
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "These metrics represent the model evaluation results "
        "obtained during the model development stage."
    )


# ============================================================
# XAI & EXPLAINABILITY
# ============================================================

elif page == "XAI & Explainability":

    col1, col2 = st.columns([0.12, 0.88])

    with col1:
        st.image(asset("brain-circuit.png"), width=65)

    with col2:
        st.header("XAI & Explainability")
        st.caption(
            "Understanding why the machine learning model makes "
            "fraud detection decisions."
        )

    st.divider()

    st.subheader("Explainability Framework")

    c1, c2 = st.columns(2)

    with c1:

        st.image(
            asset("trend-chart.png"),
            width=50
        )

        st.markdown("### Historical Explainability")

        st.write("""
The historical component analyses model behaviour across
the evaluation dataset and identifies the features that
generally contribute to fraud predictions.
""")

    with c2:

        st.image(
            asset("live-pulse.png"),
            width=50
        )

        st.markdown("### Real-Time Explainability")

        st.write("""
The real-time component explains individual transactions
as they pass through the Kafka streaming pipeline, helping
operational teams understand the basis of each prediction.
""")

    st.divider()

    st.subheader("SHAP Feature Importance")

    st.info(
        "SHAP visualisations will be connected to the trained "
        "Random Forest model in the next integration stage."
    )


# ============================================================
# TRANSACTION EXPLORER
# ============================================================

elif page == "Transaction Explorer":

    col1, col2 = st.columns([0.12, 0.88])

    with col1:
        st.image(asset("search-document.png"), width=65)

    with col2:
        st.header("Transaction Explorer")
        st.caption(
            "Inspect individual transactions and model decisions."
        )

    st.divider()

    transaction_id = st.text_input(
        "Transaction ID",
        placeholder="Enter transaction ID..."
    )

    if transaction_id:

        st.subheader("Transaction Details")

        st.write(
            f"Transaction ID: **{transaction_id}**"
        )

        st.info(
            "Transaction records will be retrieved from the "
            "real-time processing store in the next integration stage."
        )

    else:

        st.info(
            "Enter a transaction ID to inspect a transaction."
        )


# ============================================================
# MONITORING & TRENDS
# ============================================================

elif page == "Monitoring & Trends":

    col1, col2 = st.columns([0.12, 0.88])

    with col1:
        st.image(asset("trend-chart.png"), width=65)

    with col2:
        st.header("Monitoring & Trends")
        st.caption(
            "Monitor transaction and fraud patterns over time."
        )

    st.divider()

    st.subheader("Transaction Activity")

    st.info(
        "Real-time transaction trend visualisations will be "
        "connected to the Kafka stream."
    )

    st.subheader("Fraud Trend")

    st.info(
        "Historical and live fraud trends will be displayed here."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    col1, col2 = st.columns([0.12, 0.88])

    with col1:
        st.image(asset("info-circle.png"), width=65)

    with col2:
        st.header("About the System")

    st.divider()

    st.markdown("""
### Development of Adaptive and Interpretable Machine Learning
### Framework for Real-Time Fraud Detection and Operational
### Decision Support in Digital Banking

**Developed by:**  
**Adeseye Samuel Ademola**

### Core Technologies

- Python
- Scikit-learn
- Random Forest
- SHAP
- Apache Kafka
- Pandas
- NumPy
- Streamlit

### System Architecture

The system integrates historical machine learning analysis
with real-time transaction streaming and explainable AI.

The objective is not only to identify potentially fraudulent
transactions, but also to provide interpretable evidence that
can assist operational teams in making informed decisions.
""")

    st.divider()

    st.caption(
        "Fraud Detection & Operational Decision Support System"
    )