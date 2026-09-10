import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

def render_executive_overview():

    # --------------------------------------------------------
    # CURRENT PROJECT METRICS
    # --------------------------------------------------------

    TOTAL_TRANSACTIONS = 284807
    FRAUD_CASES = 492
    LEGITIMATE_CASES = TOTAL_TRANSACTIONS - FRAUD_CASES

    FRAUD_RATE = (FRAUD_CASES / TOTAL_TRANSACTIONS) * 100

    MODEL_ACCURACY = 100.0
    MODEL_PRECISION = 90.0
    MODEL_RECALL = 77.0
    MODEL_F1 = 83.0

    ROC_AUC = 96.06
    PR_AUC = 81.27

    # --------------------------------------------------------
    # PAGE STYLE
    # --------------------------------------------------------

    st.markdown("""
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 3px;
    }

    .subtitle {
        color: #8b8b96;
        font-size: 14px;
        margin-bottom: 25px;
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

    .section-title {
        font-size: 17px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 8px;
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

    </style>
    """, unsafe_allow_html=True)


    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="main-title">Executive Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Fraud Detection & Operational Decision Support System'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # KPI ROW
    # ========================================================

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Transactions</div>
            <div class="kpi-value">{TOTAL_TRANSACTIONS:,}</div>
            <div class="kpi-description">Historical transactions analysed</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Fraud Cases</div>
            <div class="kpi-value">{FRAUD_CASES:,}</div>
            <div class="kpi-description">Detected fraud transactions</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Fraud Rate</div>
            <div class="kpi-value">{FRAUD_RATE:.2f}%</div>
            <div class="kpi-description">Fraud proportion in dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">ROC-AUC</div>
            <div class="kpi-value">{ROC_AUC:.2f}%</div>
            <div class="kpi-description">Discrimination performance</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">PR-AUC</div>
            <div class="kpi-value">{PR_AUC:.2f}%</div>
            <div class="kpi-description">Fraud-class performance</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # CHART ROW 1
    # ========================================================

    col1, col2 = st.columns([1, 1])


    # --------------------------------------------------------
    # FRAUD DISTRIBUTION
    # --------------------------------------------------------

    with col1:

        st.markdown(
            '<div class="section-title">Transaction Risk Distribution</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Legitimate", "Fraud"],
                    values=[LEGITIMATE_CASES, FRAUD_CASES],
                    hole=0.68,
                    textinfo="label+percent",
                    textposition="outside"
                )
            ]
        )

        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            legend=dict(
                orientation="h",
                y=-0.05
            ),
            annotations=[
                dict(
                    text=f"{FRAUD_RATE:.2f}%<br>Fraud",
                    x=0.5,
                    y=0.5,
                    font=dict(
                        size=18,
                        color="white"
                    ),
                    showarrow=False
                )
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    with col2:

        st.markdown(
            '<div class="section-title">Fraud Detection Model Performance</div>',
            unsafe_allow_html=True
        )

        metrics = [
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "PR-AUC"
        ]

        values = [
            MODEL_PRECISION,
            MODEL_RECALL,
            MODEL_F1,
            ROC_AUC,
            PR_AUC
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=values,
                y=metrics,
                orientation="h",
                text=[f"{v:.1f}%" for v in values],
                textposition="outside"
            )
        )

        fig.update_layout(
            height=320,
            xaxis=dict(
                range=[0, 110],
                title="Performance (%)"
            ),
            yaxis=dict(
                autorange="reversed"
            ),
            margin=dict(l=10, r=40, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )


    # ========================================================
    # CHART ROW 2
    # ========================================================

    col3, col4 = st.columns([1, 1])


    # --------------------------------------------------------
    # MODEL KPI SUMMARY
    # --------------------------------------------------------

    with col3:

        st.markdown(
            '<div class="section-title">Model Evaluation Summary</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="status-card">

            <div class="status-row">
                <span>Accuracy</span>
                <span class="status-good">{MODEL_ACCURACY:.2f}%</span>
            </div>

            <div class="status-row">
                <span>Precision</span>
                <span class="status-good">{MODEL_PRECISION:.2f}%</span>
            </div>

            <div class="status-row">
                <span>Recall</span>
                <span class="status-good">{MODEL_RECALL:.2f}%</span>
            </div>

            <div class="status-row">
                <span>F1-Score</span>
                <span class="status-good">{MODEL_F1:.2f}%</span>
            </div>

            <div class="status-row">
                <span>ROC-AUC</span>
                <span class="status-good">{ROC_AUC:.2f}%</span>
            </div>

            <div class="status-row">
                <span>PR-AUC</span>
                <span class="status-good">{PR_AUC:.2f}%</span>
            </div>

        </div>
        """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # SYSTEM PIPELINE STATUS
    # --------------------------------------------------------

    with col4:

        st.markdown(
            '<div class="section-title">Decision Support Pipeline</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="status-card">

            <div class="status-row">
                <span>Transaction Ingestion</span>
                <span class="status-good">● ACTIVE</span>
            </div>

            <div class="status-row">
                <span>Kafka Event Streaming</span>
                <span class="status-good">● ACTIVE</span>
            </div>

            <div class="status-row">
                <span>Apache Spark Processing</span>
                <span class="status-good">● ACTIVE</span>
            </div>

            <div class="status-row">
                <span>Fraud Classification</span>
                <span class="status-good">● ACTIVE</span>
            </div>

            <div class="status-row">
                <span>Explainability Layer</span>
                <span class="status-good">● ACTIVE</span>
            </div>

            <div class="status-row">
                <span>Decision Support Dashboard</span>
                <span class="status-good">● ACTIVE</span>
            </div>

        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # EXECUTIVE INSIGHTS
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Executive Insights</div>',
        unsafe_allow_html=True
    )

    i1, i2, i3 = st.columns(3)


    with i1:

        st.markdown(f"""
        <div class="insight-card">

            <div class="insight-title">
                Fraud Exposure
            </div>

            <div class="insight-text">
                The historical dataset contains
                <b>{FRAUD_CASES:,}</b> identified fraudulent
                transactions out of <b>{TOTAL_TRANSACTIONS:,}</b>
                transactions analysed.
                This represents a fraud rate of
                <b>{FRAUD_RATE:.2f}%</b>.
            </div>

        </div>
        """, unsafe_allow_html=True)


    with i2:

        st.markdown(f"""
        <div class="insight-card">

            <div class="insight-title">
                Detection Capability
            </div>

            <div class="insight-text">
                The Random Forest model achieved a
                ROC-AUC of <b>{ROC_AUC:.2f}%</b> and
                PR-AUC of <b>{PR_AUC:.2f}%</b>,
                demonstrating strong ability to distinguish
                fraudulent transactions from legitimate activity.
            </div>

        </div>
        """, unsafe_allow_html=True)


    with i3:

        st.markdown(f"""
        <div class="insight-card">

            <div class="insight-title">
                Operational Consideration
            </div>

            <div class="insight-text">
                Fraud recall is currently
                <b>{MODEL_RECALL:.1f}%</b>, meaning the system
                identifies a substantial proportion of fraudulent
                activity while maintaining a precision of
                <b>{MODEL_PRECISION:.1f}%</b>.
                This supports risk-based operational review.
            </div>

        </div>
        """, unsafe_allow_html=True)