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
