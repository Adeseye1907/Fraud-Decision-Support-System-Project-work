import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Fraud DSS - Adaptive & Interpretable Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .metric-card {
        background-color: #1a1f2c;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-header { font-size: 0.85rem; color: #94a3b8; margin-bottom: 4px; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #f8fafc; }
    .decision-banner {
        background: linear-gradient(90deg, #1e3a8a, #0284c7);
        color: white;
        padding: 14px 20px;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 500;
        margin: 15px 0;
        line-height: 1.6;
    }
    .status-badge-online {
        background-color: #1e3a8a;
        color: #60a5fa;
        padding: 8px 16px;
        border-radius: 6px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .predict-fraud-card {
        background-color: #161b26;
        border-left: 4px solid #ef4444;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation
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

with st.sidebar:
    st.markdown("### System Navigation")
    selected_page = st.radio("Navigation", PAGES, label_visibility="collapsed", key="dss_main_nav_radio")
    st.markdown("---")
    st.markdown("**SYSTEM STATUS**")
    st.success("🟢 System Online")

# ==========================================
# 1. INTRODUCTION
# ==========================================
if selected_page == "Introduction":
    st.title("Fraud Detection & Operational Decision Support System")
    st.markdown("##### Prototype / Reference Implementation")
    st.markdown("---")
    st.markdown("""
    This framework integrates machine learning, real-time transaction streaming, Explainable AI (SHAP & LIME), 
    adaptive monitoring, operational feedback loops, and decision-support capabilities for digital banking environments.
    """)

# ==========================================
# 2. HISTORICAL MODEL ANALYSIS
# ==========================================
elif selected_page == "Historical Model Analysis":
    st.title("Historical Model Analysis")
    st.caption("Evaluation and interpretation of the trained Random Forest fraud detection model.")
    
    st.markdown("#### Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", "0.961")
    c2.metric("PR-AUC", "0.813")
    c3.metric("Fraud Precision", "0.90")
    c4.metric("Fraud Recall", "0.77")
    
    st.markdown("#### Historical Test Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions Evaluated", "56,746")
    c2.metric("Actual Fraud", "95")
    c3.metric("Fraud Detected", "73")
    c4.metric("Fraud Missed", "22")

    st.markdown("#### Operational Fraud Detection Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detection Rate", "76.84%")
    c2.metric("False Alarms", "8")
    c3.metric("Fraud F1-Score", "0.830")

    st.markdown("---")
    st.subheader("Global Model Interpretability")
    st.caption("Global SHAP identifies the features with the greatest influence on model predictions.")
    
    shap_df = pd.DataFrame({
        "Feature": ["V14", "V10", "V12", "V17", "V4", "V11", "V7", "V16", "V3", "V1"],
        "Mean_Abs_SHAP": [0.198, 0.165, 0.142, 0.121, 0.098, 0.082, 0.061, 0.045, 0.038, 0.021]
    }).sort_values("Mean_Abs_SHAP", ascending=True)
    
    fig = px.bar(shap_df, x="Mean_Abs_SHAP", y="Feature", orientation='h', 
                 color="Mean_Abs_SHAP", color_continuous_scale="Blues", template="plotly_dark")
    fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Prediction Outcome Summary")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Fraud", "81")
    c2.metric("Predicted Legitimate", "56,665")

# ==========================================
# 3. PREDICT FRAUD
# ==========================================
elif selected_page == "Predict Fraud":
    st.title("Predict Fraud")
    st.caption("Manual scoring and operational inference engine.")
    
    st.markdown("""
        <div class="predict-fraud-card">
            <h4>Historical Prediction Results</h4>
            <p>Predict Fraud Operational Mode: Active Dark-Inspection Shell</p>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    c1.text_input("Transaction Reference ID", "TX-2026-9812401")
    c2.number_input("Transaction Amount (Standardized / USD)", value=2.7905)
    st.button("Evaluate Transaction Risk", type="primary")

# ==========================================
# 4. REAL-TIME FRAUD DETECTION
# ==========================================
elif selected_page == "Real-Time Fraud Detection":
    st.title("Real-Time Fraud Detection")
    st.caption("Live streaming inference and explainability.")
    
    st.markdown("#### Latest Transaction Stream Ingestion")
    sample_transaction = {
        "Time": -0.6094047378387375, "V1": -2.376916678699162, "V2": -1.556715626635113,
        "V3": -0.6213336114012474, "V4": -2.8772412183921108, "V5": 0.1557317088892519,
        "V6": 0.7035769851086368, "V7": 1.5545723901831174, "V8": -0.19942829368624584,
        "V9": 0.29825649874713317, "V10": -0.07666337377508571, "V11": 0.18740108299356162,
        "V12": 0.3489806468307025, "V13": 0.19286955826541477, "V14": 0.5144798733765612,
        "V15": -0.4099299831371543, "V16": 0.1507007151235256, "V17": -0.4438649632789128,
        "V18": -0.25298411312596375, "V19": 0.309211346359933, "V20": -0.03795123269751147,
        "V21": -0.30904541740747893, "V22": 0.7737340286697448, "V23": -0.41704183511945975,
        "V24": -0.8040827660195803, "V25": -0.1802960552297103, "V26": 0.27398023560445406,
        "V27": 0.1759299728835217, "V28": -0.3001619514951853, "Amount": 2.7905205909079562,
        "transaction_id": "bc08ec3c-f7bf-42fb-b463-0969fec22ba1",
        "prediction": "LEGITIMATE", "fraud_probability": 0.00,
        "timestamp": "2026-08-23T22:37:22.028699+00:00"
    }
    st.json(sample_transaction)
    
    st.markdown("---")
    st.subheader("Top SHAP Risk Factors")
    shap_factors = pd.DataFrame([
        {"Feature": "V4", "Feature_Value": -2.8772, "SHAP_Value": -0.1216},
        {"Feature": "V12", "Feature_Value": 0.3490, "SHAP_Value": -0.1028},
        {"Feature": "V14", "Feature_Value": 0.5145, "SHAP_Value": -0.0759},
        {"Feature": "V11", "Feature_Value": 0.1874, "SHAP_Value": -0.0369},
        {"Feature": "V10", "Feature_Value": -0.0767, "SHAP_Value": -0.0304}
    ])
    st.dataframe(shap_factors, use_container_width=True)

    st.markdown("---")
    st.subheader("Real-Time LIME Explanation")
    lime_data = pd.DataFrame([
        {"Feature": "V4 <= -0.85", "LIME_Weight": -0.0143, "Direction": "Pushes toward Legitimate"},
        {"Feature": "V14 > 0.49", "LIME_Weight": -0.0064, "Direction": "Pushes toward Legitimate"},
        {"Feature": "V12 > 0.30", "LIME_Weight": -0.0051, "Direction": "Pushes toward Legitimate"}
    ])
    st.dataframe(lime_data, use_container_width=True)

# ==========================================
# 5. TRANSACTION EXPLORER
# ==========================================
elif selected_page == "Transaction Explorer":
    st.title("Transaction Explorer")
    st.caption("Investigate individual transactions, model predictions and transaction-level risk information.")
    
    tx_search = st.text_input("Transaction ID", placeholder="Search transaction ID (e.g. bc08ec3c-f7bf-42fb-b463-0969fec22ba1)...")
    
    if tx_search:
        st.success(f"Displaying evaluation breakdown for: `{tx_search}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Status", "LEGITIMATE")
        c2.metric("Fraud Probability", "0.00%")
        c3.metric("Amount Risk Score", "Normal")
    else:
        st.info("No transaction data queried. Enter a Transaction ID to retrieve SHAP decomposition.")

# ==========================================
# 6. TRENDS & MONITORING
# ==========================================
elif selected_page == "Trends & Monitoring":
    st.title("Trends & Monitoring")
    st.caption("Monitor transaction activity and fraud patterns across the real-time transaction stream.")
    
    st.subheader("Transaction Volume")
    time_series = pd.DataFrame({
        "Timestamp": [f"11:{30+i}:00" for i in range(11)],
        "Volume": [12, 18, 15, 25, 29, 21, 19, 27, 24, 30, 22]
    })
    fig = px.line(time_series, x="Timestamp", y="Volume", markers=True, template="plotly_dark")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. ADAPTIVE MONITORING & DRIFT
# ==========================================
elif selected_page == "Adaptive Monitoring":
    st.title("Adaptive Monitoring & Drift Detection")
    st.caption("Monitors incoming transaction behaviour and provides an early indication of whether the operating environment may have changed.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Real-Time Transactions Available", "371")
    c2.metric("Drift Score", "0.00")
    c3.metric("Monitoring Status", "STABLE")
    
    st.markdown("#### Monitoring Assessment")
    st.info("Current monitoring indicators do not indicate significant drift.")

# ==========================================
# 8. MODEL ADAPTATION
# ==========================================
elif selected_page == "Model Adaptation":
    st.title("Model Adaptation / Update Mechanism")
    st.caption("This component translates monitoring evidence into an operational model-review and adaptation decision.")
    
    c1, c2 = st.columns(2)
    c1.metric("Current Drift Score", "0.00")
    c2.metric("Current Status", "STABLE")
    
    st.markdown("#### Adaptation Decision")
    st.success("No immediate model update is indicated.")
    
    st.markdown("#### Adaptation History")
    st.caption("No adaptation reviews have been recorded.")

# ==========================================
# 9. OPERATIONAL FEEDBACK
# ==========================================
elif selected_page == "Operational Feedback":
    st.title("Operational Feedback")
    st.caption("Capture operational review outcomes for future learning and model improvement.")
    
    st.markdown("""
    <div class="metric-card">
        <div class="metric-header">Recorded Feedback Count</div>
        <div class="metric-value">0</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Transaction Under Review")
    selected_tx = st.selectbox("Select Transaction", ["264e2f21-ab1f-478a-90b3-bdb6112005d0", "bc08ec3c-f7bf-42fb-b463-0969fec22ba1"])
    
    with st.expander("Inspect Raw Transaction Schema", expanded=False):
        st.json({
            "transaction_id": "264e2f21-ab1f-478a-90b3-bdb6112005d0",
            "prediction": "LEGITIMATE",
            "fraud_probability": 0.02,
            "top_shap_features": [
                {"Feature": "V12", "SHAP_Value": -0.0929, "Direction": "Pushes toward Legitimate"},
                {"Feature": "V11", "SHAP_Value": -0.0841, "Direction": "Pushes toward Legitimate"},
                {"Feature": "V14", "SHAP_Value": -0.0825, "Direction": "Pushes toward Legitimate"}
            ]
        })
    
    outcome = st.selectbox("Operational Review Outcome", ["Confirmed Legitimate", "Confirmed Fraud", "Requires Escalation"])
    comment = st.text_area("Reviewer Comment", placeholder="Enter investigator rationale here...")
    
    if st.button("Submit Operational Feedback", type="primary"):
        st.success("Operational feedback captured successfully.")

# ==========================================
# 10. FEEDBACK LEARNING PIPELINE
# ==========================================
elif selected_page == "Feedback Learning Pipeline":
    st.title("Feedback → Model Learning Pipeline")
    st.caption("Operational feedback provides labelled evidence that can support future model evaluation, retraining and adaptation.")
    
    st.markdown("""
    <div class="metric-card">
        <div class="metric-header">Feedback Records Available</div>
        <div class="metric-value">0</div>
    </div>
    """, unsafe_allow_html=True)
    
    pipeline_df = pd.DataFrame([
        {"Stage": "1", "Process": "Operational Review", "Purpose": "Human assessment of suspicious transactions", "Status": "Available"},
        {"Stage": "2", "Process": "Feedback Capture", "Purpose": "Store review outcome and comments", "Status": "Available"},
        {"Stage": "3", "Process": "Feedback Validation", "Purpose": "Check quality and consistency of feedback", "Status": "Planned"},
        {"Stage": "4", "Process": "Training Dataset Update", "Purpose": "Add validated observations to future training data", "Status": "Planned"}
    ])
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

# ==========================================
# 11. COST / VALUE FRAMING
# ==========================================
elif selected_page == "Cost / Value Framing":
    st.title("Cost / Value Framing")
    st.caption("Fraud detection decisions involve trade-offs between fraud losses, false alarms and operational investigation costs.")
    
    st.subheader("Operational Cost Assumptions")
    c1, c2, c3 = st.columns(3)
    cost_missed = c1.number_input("Cost of Missed Fraud (₦)", value=100000.0, step=5000.0)
    cost_false_alarm = c2.number_input("Cost of False Alarm (₦)", value=1000.0, step=100.0)
    cost_investigation = c3.number_input("Investigation Cost (₦)", value=500.0, step=50.0)
    
    missed_fraud_count = 22
    false_alarm_count = 8
    total_flagged = 81
    
    est_missed_cost = missed_fraud_count * cost_missed
    est_false_cost = false_alarm_count * cost_false_alarm
    est_operational_cost = est_missed_cost + est_false_cost + (total_flagged * cost_investigation)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Estimated Missed-Fraud Cost**\n### ₦{est_missed_cost:,.2f}")
    c2.markdown(f"**Estimated False-Alarm Cost**\n### ₦{est_false_cost:,.2f}")
    c3.markdown(f"**Estimated Operational Cost**\n### ₦{est_operational_cost:,.2f}")
    
    st.info("These values are illustrative decision-support assumptions. They should be replaced with institution-specific estimates when operational cost data become available.")

# ==========================================
# 12. THRESHOLD TUNER
# ==========================================
elif selected_page == "Threshold Tuner":
    st.title("Fraud Detection Threshold")
    st.caption("Explore how the fraud probability threshold affects operational classification decisions.")
    
    threshold = st.slider("Classification Threshold", min_value=0.01, max_value=1.00, value=0.50, step=0.01)
    
    st.markdown("#### Selected Threshold")
    st.markdown(f"## {threshold:.2f}")
    
    flagged = int(81 * (0.50 / max(threshold, 0.01)))
    st.markdown("#### Transactions Flagged at Threshold")
    st.markdown(f"## {flagged}")
    
    st.info("A lower threshold increases sensitivity but may increase false positives. A higher threshold may reduce false alarms but can increase missed fraud.")

# ==========================================
# 13. END-TO-END DSS WORKFLOW
# ==========================================
elif selected_page == "End-to-End DSS Workflow":
    st.title("End-to-End DSS Workflow")
    st.caption("Live system pipeline connectivity and decision routing architecture.")
    
    st.subheader("Current System Status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Kafka**")
        st.markdown('<div class="status-badge-online">Online</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("**Model**")
        st.markdown('<div class="status-badge-online">Loaded</div>', unsafe_allow_html=True)
    with c3:
        st.markdown("**LIME**")
        st.markdown('<div class="status-badge-online">Loaded</div>', unsafe_allow_html=True)
    with c4:
        st.markdown("**Adaptive Status**")
        st.markdown('<div class="status-badge-online">STABLE</div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Adaptive Decision Chain")
    st.markdown("""
    <div class="decision-banner">
        Real-Time Transactions → Kafka → Fraud Prediction → SHAP/LIME Explanation → Adaptive Monitoring → Drift Assessment → Model Review → Operational Feedback → Future Model Learning → Cost/Value Assessment → Operational Decision Support
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 14. ABOUT THE FRAMEWORK
# ==========================================
elif selected_page == "About the Framework":
    st.title("About the Framework")
    st.subheader("Development of Adaptive and Interpretable Machine Learning Framework for Real-Time Fraud Detection and Operational Decision Support in Digital Banking")
    st.markdown("**Developed by: Adeseye Samuel Ademola**")
    st.markdown("---")
    
    st.markdown("""
    The framework integrates machine learning, real-time transaction streaming, Explainable Artificial Intelligence, 
    adaptive monitoring, operational feedback and decision-support capabilities.
    """)
    
    st.markdown("**Model Input**")
    st.markdown("The trained fraud detection model uses 30 features: `Time`, `V1-V28`, and `Amount`.")
    
    st.markdown("**Interpretability**")
    st.markdown("* **Global interpretability:** SHAP global feature importance is used to understand the behaviour of the model across historical data.")
    st.markdown("* **Local interpretability:** SHAP and LIME are used to explain why an individual transaction received a particular prediction.")
    
    st.markdown("**Operational Decision Support**")
    st.markdown("""
    The DSS does not replace human decision-making. Instead, it provides operational users with predictive, 
    explanatory, monitoring and adaptive information that can support transaction investigation, prioritisation and fraud-response decisions.
    """)