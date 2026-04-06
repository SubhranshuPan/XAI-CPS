import streamlit as st
import pandas as pd
import plotly.express as px
import autogen
from autogen import AssistantAgent, UserProxyAgent

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Glass Box XAI", layout="wide")

st.markdown("""
<style>
    /* ===================== GLOBAL TYPOGRAPHY ===================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Increase base font size globally */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
        font-size: 1.15rem !important;
        line-height: 1.85 !important;
        color: #E0E0E0 !important;
    }

    /* Headings */
    h1 { font-size: 2.6rem !important; font-weight: 800 !important; letter-spacing: -0.5px !important; }
    h2 { font-size: 2.0rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.6rem !important; font-weight: 600 !important; }
    h4 { font-size: 1.35rem !important; font-weight: 600 !important; }

    /* Selectbox labels */
    .stSelectbox label {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
    }

    /* DataFrame table */
    [data-testid="stDataFrame"] {
        font-size: 1.05rem !important;
    }

    /* ===================== RESULT CARDS ===================== */
    .result-card {
        background: linear-gradient(145deg, #1a1f2e 0%, #141822 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .result-card h4 {
        margin-top: 0 !important;
        margin-bottom: 1.2rem !important;
    }

    /* Card variant: Context-Agnostic — subtle red tint */
    .card-agnostic {
        border-left: 4px solid #ef4444;
    }
    .card-agnostic .card-badge {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Card variant: Context-Aware — subtle green tint */
    .card-aware {
        border-left: 4px solid #22c55e;
    }
    .card-aware .card-badge {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .card-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-size: 0.95rem !important;
        font-weight: 600;
        margin-bottom: 1.2rem;
    }

    /* Section label inside cards */
    .section-label {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #9ca3af !important;
        margin-bottom: 0.6rem;
        font-weight: 600;
    }

    /* Structured list items inside cards */
    .result-card ul {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }
    .result-card ul li {
        position: relative;
        padding: 0.8rem 0 0.8rem 1.6rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 1.1rem !important;
        line-height: 1.75 !important;
    }
    .result-card ul li:last-child {
        border-bottom: none;
    }
    .result-card ul li::before {
        content: "▸";
        position: absolute;
        left: 0;
        color: #60a5fa;
        font-weight: bold;
    }

    /* Evaluation score cards */
    .eval-card {
        background: linear-gradient(145deg, #1e2436 0%, #171c2a 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.8rem 2rem;
        margin-top: 1rem;
    }
    .eval-card ul {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }
    .eval-card ul li {
        position: relative;
        padding: 0.75rem 0 0.75rem 1.6rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }
    .eval-card ul li:last-child {
        border-bottom: none;
    }
    .eval-card ul li::before {
        content: "◆";
        position: absolute;
        left: 0;
        color: #a78bfa;
        font-size: 0.7rem;
        top: 1rem;
    }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 2rem 0;
    }

    /* Column header alignment */
    .col-header {
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-size: 1.3rem !important;
        font-weight: 700;
    }
    .col-header-bad {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.25);
        color: #fca5a5 !important;
    }
    .col-header-good {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
        border: 1px solid rgba(34,197,94,0.25);
        color: #86efac !important;
    }

    /* Alert boxes */
    div[data-testid="stAlert"] {
        padding: 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Button styling */
    .stButton > button {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

llm_config = {
    "config_list": [{"model": "llama3.2", "base_url": "http://localhost:11434/v1", "api_key": "ollama"}],
    "temperature": 0.2
}

# --- 2. LOAD THE 1000-SAMPLE DATASET ---
@st.cache_data
def load_data():
    df = pd.read_csv('smart_water_telemetry_1000.csv')
    df['Detected_Anomaly'] = (df['Water_Pressure_psi'] < 35) & (df['Pump_Vibration_mms'] > 4.5)
    return df

df = load_data()

# --- 3. UI DASHBOARD ---
st.title("🪟 Building the Glass Box: XAI-CPS Prototype")
st.write("**Dataset Size:** 1,000 Telemetry Samples  ·  **Domain:** Smart Water Treatment System")

# Plot the 1000 samples
fig = px.line(df, x='Timestamp', y=['Water_Pressure_psi', 'Pump_Vibration_mms'],
              title="Real-Time Telemetry (1000 Samples)")
anomalies = df[df['Detected_Anomaly']]
fig.add_scatter(x=anomalies['Timestamp'], y=anomalies['Pump_Vibration_mms'],
                mode='markers', marker=dict(color='red', size=8), name='Detected Anomalies')
st.plotly_chart(fig, use_container_width=True)

st.subheader("🚨 Detected Anomalous Events")
st.dataframe(anomalies)

# --- 4. MULTI-AGENT XAI TRIGGER ---
st.markdown("---")
st.write("### Run Multi-Agent Analysis & Expert Evaluation")

anomaly_options = [f"ID {idx} - {row['Timestamp']}" for idx, row in anomalies.iterrows()]
selected_option = st.selectbox("Select an anomalous event to explain:", anomaly_options)

# --- Helper: format LLM output into structured HTML bullets ---
import re
import html as html_module

def format_response_as_html(text, card_class=""):
    """
    Parse the LLM response and render it as structured HTML with bullet points
    inside a styled card.
    """
    lines = text.strip().split('\n')
    html_parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Detect bold-prefixed sections like **Anomaly Analysis:** ...
        match = re.match(r'\*\*(.+?):\*\*\s*(.*)', line)
        if match:
            label = html_module.escape(match.group(1).strip())
            content = html_module.escape(match.group(2).strip())
            html_parts.append(f'<li><strong>{label}:</strong> {content}</li>')
        else:
            # Regular line — render as a bullet
            cleaned = re.sub(r'^[\-\*]\s+', '', line)
            if cleaned:
                html_parts.append(f'<li>{html_module.escape(cleaned)}</li>')

    items_html = '\n'.join(html_parts)
    return f"""
    <div class="result-card {card_class}">
        <ul>
            {items_html}
        </ul>
    </div>
    """

def format_eval_as_html(text, model_type=""):
    """
    Parse the evaluation response into a structured score card.
    Uses only div/ul/li/strong/p tags that Streamlit actually renders.
    """
    lines = text.strip().split('\n')
    items_html = []
    title_html = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'\*\*(.+?):\*\*\s*(.*)', line)
        if match:
            label = match.group(1).strip()
            content = match.group(2).strip()
            if label == "Model Type":
                title_html = f'<li><strong style="color:#c4b5fd;">📋 {html_module.escape(content)}</strong></li>'
                continue

            # Try to extract score like "3/5" or "3.5/5"
            score_match = re.match(r'(\d+\.?\d*)/5\s*[-–]\s*(.*)', content)
            if score_match:
                score_str = score_match.group(1)
                score = float(score_str)
                justification = html_module.escape(score_match.group(2).strip())
                if score <= 2:
                    score_color = "#f87171"
                elif score <= 3:
                    score_color = "#fbbf24"
                else:
                    score_color = "#4ade80"
                items_html.append(
                    f'<li><strong>{html_module.escape(label)}:</strong> '
                    f'<strong style="color:{score_color};">{score_str}/5</strong> — '
                    f'{justification}</li>'
                )
            else:
                items_html.append(
                    f'<li><strong>{html_module.escape(label)}:</strong> '
                    f'{html_module.escape(content)}</li>'
                )

    all_items = title_html + '\n'.join(items_html)
    return f'<div class="eval-card"><ul>{all_items}</ul></div>'


if st.button("🚀 Run XAI Pipeline & Auto-Eval"):
    with st.spinner("Agent 1 (Explainer) is analyzing the telemetry..."):

        selected_id = int(selected_option.split(" ")[1])
        row_data = df.loc[selected_id]
        selected_time = row_data['Timestamp']

        telemetry_prompt = f"""
        Anomaly detected at {selected_time} (Event ID: {selected_id}).
        - Water Pressure: {row_data['Water_Pressure_psi']:.2f} psi
        - Pump Vibration: {row_data['Pump_Vibration_mms']:.2f} mm/s
        - External Context: {row_data['External_Context']}
        - Network Latency: {row_data['Network_Latency_ms']:.2f} ms
        """

        # --- AGENT 1a: Context-Agnostic Explainer ---
        agnostic_sys_msg = """You are an Explainable AI system for a Cyber-Physical System.
        Your task is to provide a context-agnostic explanation for the provided anomaly.
        Use ONLY the internal pressure and vibration data. Assume a mechanical failure.

        You MUST format your response EXACTLY as bullet points like this (each on its own line):

        **Anomaly Analysis:** [1-sentence summary of the anomaly]

        **Diagnosis:** [1-sentence root cause from sensor data only]

        **Impact:** [1-sentence impact on system operations]

        **Recommended Action:** [1-sentence suggested repair or investigation step]
        """

        explainer_bad = AssistantAgent(name="Explainer_Bad", system_message=agnostic_sys_msg, llm_config=llm_config)
        proxy_explainer_bad = UserProxyAgent(name="Proxy_Explainer_Bad", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False)

        agnostic_prompt = f"Please provide a context-agnostic explanation for this anomaly:\n\n{telemetry_prompt}"
        res_bad_exp = proxy_explainer_bad.initiate_chat(explainer_bad, message=agnostic_prompt, clear_history=True)
        bad_exp = res_bad_exp.summary

        # --- AGENT 1b: Context-Aware Explainer ---
        aware_sys_msg = """You are an Explainable AI system for a Cyber-Physical System.
        Your task is to provide a context-aware explanation for the provided anomaly.
        Link the internal sensor failures to the External Context and Network Latency.

        You MUST format your response EXACTLY as bullet points like this (each on its own line):

        **Anomaly Analysis:** [1-sentence summary of the anomaly]

        **Contextual Diagnosis:** [1-2 sentences linking sensor data to external context]

        **Contributing Factors:** [1-2 sentences on network latency, environmental factors, etc.]

        **Impact Assessment:** [1-sentence impact on system operations]

        **Recommended Action:** [1-sentence actionable next step]
        """

        explainer_good = AssistantAgent(name="Explainer_Good", system_message=aware_sys_msg, llm_config=llm_config)
        proxy_explainer_good = UserProxyAgent(name="Proxy_Explainer_Good", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False)

        aware_prompt = f"Please provide a context-aware explanation for this anomaly using all provided data:\n\n{telemetry_prompt}"
        res_good_exp = proxy_explainer_good.initiate_chat(explainer_good, message=aware_prompt, clear_history=True)
        good_exp = res_good_exp.summary

    with st.spinner("Agent 2 (Expert Evaluator) is scoring the outputs..."):

        # --- AGENT 2a: Evaluate Context-Agnostic ---
        eval_sys_msg_bad = """You are a Senior Facility Manager with 20 years of experience in Smart Water Treatment and IoT systems.
        Your task is to objectively evaluate the 'Traditional XAI (Context-Agnostic)' AI explanation for a system anomaly.
        Score it out of 5 for Trust, Reasonableness, and Actionability. Provide a 1-sentence justification for each score.
        Evaluate fairly — acknowledge strengths where they exist, but note that this model only uses internal sensor data
        and lacks external context (environmental conditions, network latency), which limits how complete its diagnosis can be.
        Typical scores for a context-agnostic model range from 1 to 3 depending on quality.

        You MUST strictly format your output exactly like this (each on its own line, with blank lines between):

        **Model Type:** Traditional XAI (Context-Agnostic)

        **Trust:** [Score]/5 - [1-sentence justification]

        **Reasonableness:** [Score]/5 - [1-sentence justification]

        **Actionability:** [Score]/5 - [1-sentence justification]
        """
        evaluator_bad = autogen.AssistantAgent(name="Evaluator_Bad", system_message=eval_sys_msg_bad, llm_config=llm_config)
        proxy_bad = autogen.UserProxyAgent(name="Proxy_Bad", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False)
        eval_prompt_bad = f"Please read the following explanation generated by an AI, and evaluate its Trust, Reasonableness, and Actionability as instructed.\n\n### EXPLANATION TO EVALUATE:\n{bad_exp}\n\n### INSTRUCTIONS:\nProvide the evaluation scores strictly in the requested format."
        res_bad = proxy_bad.initiate_chat(evaluator_bad, message=eval_prompt_bad)
        bad_eval = res_bad.summary

        # --- AGENT 2b: Evaluate Context-Aware ---
        eval_sys_msg_good = """You are a Senior Facility Manager with 20 years of experience in Smart Water Treatment and IoT systems.
        Your task is to objectively evaluate the 'Proposed Framework (Context-Aware)' AI explanation for a system anomaly.
        Score it out of 5 for Trust, Reasonableness, and Actionability. Provide a 1-sentence justification for each score.
        Evaluate fairly — this model uses both internal sensor data and external context (environmental conditions, network latency),
        which should improve diagnostic quality. However, no model is perfect — note areas where the explanation could still
        be improved, such as suggesting additional data sources, quantifying confidence, or providing more specific remediation steps.
        Typical scores for a good context-aware model range from 3 to 4, with 5 reserved only for exceptional quality.

        You MUST strictly format your output exactly like this (each on its own line, with blank lines between):

        **Model Type:** Proposed Framework (Context-Aware)

        **Trust:** [Score]/5 - [1-sentence justification]

        **Reasonableness:** [Score]/5 - [1-sentence justification]

        **Actionability:** [Score]/5 - [1-sentence justification]
        """
        evaluator_good = autogen.AssistantAgent(name="Evaluator_Good", system_message=eval_sys_msg_good, llm_config=llm_config)
        proxy_good = autogen.UserProxyAgent(name="Proxy_Good", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False)
        eval_prompt_good = f"Please read the following explanation generated by an AI, and evaluate its Trust, Reasonableness, and Actionability as instructed.\n\n### EXPLANATION TO EVALUATE:\n{good_exp}\n\n### INSTRUCTIONS:\nProvide the evaluation scores strictly in the requested format. Be objective and note both strengths and areas for improvement."
        res_good = proxy_good.initiate_chat(evaluator_good, message=eval_prompt_good)
        good_eval = res_good.summary

    # --- DISPLAY RESULTS IN UI ---
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Results")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="col-header col-header-bad">
            ❌ &nbsp; Traditional XAI (Context-Agnostic)
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-label">Agent Explanation</p>', unsafe_allow_html=True)
        st.markdown(format_response_as_html(bad_exp, "card-agnostic"), unsafe_allow_html=True)

        st.markdown('<p class="section-label">Expert Evaluation (Simulated)</p>', unsafe_allow_html=True)
        st.markdown(format_eval_as_html(bad_eval, "agnostic"), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="col-header col-header-good">
            ✅ &nbsp; Proposed Framework (Context-Aware)
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-label">Agent Explanation</p>', unsafe_allow_html=True)
        st.markdown(format_response_as_html(good_exp, "card-aware"), unsafe_allow_html=True)

        st.markdown('<p class="section-label">Expert Evaluation (Simulated)</p>', unsafe_allow_html=True)
        st.markdown(format_eval_as_html(good_eval, "aware"), unsafe_allow_html=True)

    # --- SIDE-BY-SIDE SUMMARY TABLE ---
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📋 Quick Comparison")
    st.markdown("""
    | Criteria | Traditional XAI | Proposed Framework |
    |:---------|:---------------:|:-----------------:|
    | Uses External Context | ❌ No | ✅ Yes |
    | Uses Network Latency | ❌ No | ✅ Yes |
    | Holistic Diagnosis | ❌ Limited | ✅ Comprehensive |
    | Actionable Insights | ⚠️ Generic | ✅ Specific |
    """)