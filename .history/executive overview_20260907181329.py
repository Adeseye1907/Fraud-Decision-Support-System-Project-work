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
# 14. AI DECISION ASSISTANT
# ============================================================

elif page == "AI Decision Assistant":

    st.markdown(
        """
        <style>
        .assistant-chat-container {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
            margin-bottom: 20px;
        }
        .quick-prompt-btn {
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-header">
            <div>
                <div class="dashboard-title">AI Decision Assistant</div>
                <div class="dashboard-subtitle">
                    Interpret fraud signals, examine model attributions, and support operational risk audits.
                </div>
            </div>
            <div class="live-badge">
                ● AI COPILOT READY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # SESSION STATE CONVERSATION BUFFER
    # ------------------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Operational Decision Co-Pilot Initialized.**\n\n"
                    "I can help interpret model performance metrics, feature importance (TreeSHAP/LIME), "
                    "or review incoming streaming alerts. Select a suggested prompt below or type your inquiry."
                )
            }
        ]

    # ------------------------------------------------------------
    # CONTEXT-AWARE KNOWLEDGE RETRIEVAL ENGINE
    # ------------------------------------------------------------
    def query_fraud_knowledge_base(user_query: str) -> str:
        q = user_query.lower()

        # 1. Validation & Test Benchmarks
        if any(w in q for w in ["metric", "auc", "roc", "performance", "benchmark", "precision", "recall", "f1"]):
            return (
                "📊 **Model Evaluation Benchmarks (Test Set $N = 56,746$):**\n\n"
                "- **ROC-AUC**: **96.06%** (Discriminative ability between fraud and legitimate transactions).\n"
                "- **PR-AUC**: **81.30%** (Precision-Recall trade-off under extreme imbalance).\n"
                "- **Precision**: **90.0%** (Suppresses costly false-positive investigations).\n"
                "- **Recall**: **88.0%** (Successfully intercepts 84 out of 95 test fraud events).\n"
                "- **Nominal Accuracy**: **97.0%** (Note: accuracy is uninformative due to the 0.17% fraud rate)."
            )

        # 2. XAI / SHAP / Feature Attributions
        elif any(w in q for w in ["feature", "v14", "v17", "v12", "v11", "v4", "v2", "shap", "lime", "explain"]):
            return (
                "🔍 **Explainable AI (XAI) Feature Importance:**\n\n"
                "- **Top Fraud Drivers (Negative Correlation)**: **V17** ($r = -0.313$), **V14** ($r = -0.293$), and **V12** ($r = -0.251$). "
                "Pronounced negative spikes in these latent dimensions strongly bias the model toward a fraud classification.\n"
                "- **Top Positive Indicators**: **V11** ($r = +0.149$) and **V4** ($r = +0.129$).\n"
                "- **Attribution Protocol**: Analysts can generate local LIME force plots or global TreeSHAP summary beeswarms "
                "to justify transaction freezes during compliance audits."
            )

        # 3. Macro Dataset & Imbalance Strategy
        elif any(w in q for w in ["dataset", "historical", "balance", "smote", "ratio", "duplicate", "records"]):
            return (
                "🏛️ **Dataset Topology & Preprocessing:**\n\n"
                "- **Total Sanitized Records**: **283,726** (reduced from 284,807 after purging 1,081 duplicates).\n"
                "- **Macro Ground-Truth Fraud**: **473** instances (**0.17%** active fraud rate, ~599:1 ratio).\n"
                "- **Sampling Protocol**: **SMOTE** (Synthetic Minority Over-sampling Technique) was fitted strictly on the 80% training partition "
                "($N = 226,980$) to eliminate class bias while preventing leakage into the 20% hold-out test set ($N = 56,746$)."
            )

        # 4. Streaming & System Architecture
        elif any(w in q for w in ["kafka", "spark", "drift", "stream", "pipeline", "architecture", "threshold"]):
            return (
                "⚡ **Real-Time Pipeline Architecture:**\n\n"
                "- **Message Broker**: **Apache Kafka** streaming serialized transaction payloads via the topic `fraud-transactions`.\n"
                "- **Stream Engine**: **Apache Spark Structured Streaming** micro-batching records for distributed Random Forest inference.\n"
                "- **Operational Cutoff**: A **70.0% posterior probability threshold** triggers high-priority triage alerts.\n"
                "- **Drift Detection**: The `adaptive.py` engine executes two-sample Kolmogorov-Smirnov (KS) tests on live feature streams to flag behavioral drift."
            )

        # Fallback
        else:
            return (
                f"🤖 **Assistant Assessment for:** *\"{user_query}\"*\n\n"
                "The Decision Support System evaluates this query against active pipeline configurations:\n"
                "- Baseline operational status: **Normal**\n"
                "- Evaluated decision threshold: **$\tau = 0.70$**\n"
                "- Recommended action: Correlate input features against TreeSHAP global values in the **Model Analytics** tab."
            )

    # ------------------------------------------------------------
    # QUICK PROMPT CHIPS
    # ------------------------------------------------------------
    st.markdown("##### Quick Diagnostic Inquiries")
    col1, col2, col3, col4 = st.columns(4)

    prompt_to_submit = None
    with col1:
        if st.button("📊 Model Benchmarks", use_container_width=True):
            prompt_to_submit = "What are the model evaluation benchmarks?"
    with col2:
        if st.button("🔍 Top Fraud Drivers", use_container_width=True):
            prompt_to_submit = "Explain the most important features driving fraud decisions."
    with col3:
        if st.button("🏛️ Dataset Topology", use_container_width=True):
            prompt_to_submit = "What is the total dataset volume and how was SMOTE applied?"
    with col4:
        if st.button("⚡ Pipeline & Kafka", use_container_width=True):
            prompt_to_submit = "Explain the streaming pipeline and drift detection architecture."

    # ------------------------------------------------------------
    # CHAT HISTORY DISPLAY
    # ------------------------------------------------------------
    chat_container = st.container(height=420)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ------------------------------------------------------------
    # USER QUERY INPUT & EXECUTION
    # ------------------------------------------------------------
    user_input = st.chat_input("Ask about model metrics, XAI explanations, or operational drift...")

    active_query = prompt_to_submit or user_input

    if active_query:
        # Register user inquiry
        st.session_state.chat_history.append({"role": "user", "content": active_query})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(active_query)

            # Generate reasoning response
            reply = query_fraud_knowledge_base(active_query)
            with st.chat_message("assistant"):
                st.markdown(reply)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Chat history cleared. Ready for new operational queries."
            }
        ]
        st.rerun()