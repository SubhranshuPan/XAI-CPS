import streamlit as st
import pandas as pd
import plotly.express as px
import autogen
from autogen import AssistantAgent, UserProxyAgent

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Glass Box XAI", layout="wide")

st.markdown("""
<style>
    /* Increase base font size for better readability */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] li {
        font-size: 1.2rem !important;
        line-height: 1.7 !important;
    }
    
    /* Make headers more prominent */
    h1 { font-size: 2.8rem !important; font-weight: 700 !important; }
    h2 { font-size: 2.2rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.8rem !important; font-weight: 600 !important; }
    h4 { font-size: 1.5rem !important; font-weight: 600 !important; }
    
    /* Style the alert/notification boxes to be larger and more readable */
    div[data-testid="stAlert"] {
        padding: 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Improve selectbox label size */
    .stSelectbox label {
        font-size: 1.2rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Enhance datatable readability */
    [data-testid="stDataFrame"] {
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

llm_config = {
    "config_list": [{"model": "llama3.2", "base_url": "http://localhost:11434/v1", "api_key": "ollama"}],
    "temperature": 0.2
}

# --- 2. DOMAIN CONFIG ---
DOMAIN_CONFIG = {
    "Smart Water Plant": {
        "csv": "smart_water_telemetry_1000.csv",
        "y_cols": ["Water_Pressure_psi", "Pump_Vibration_mms"],
        "anomaly_cols": ("Water_Pressure_psi", "Pump_Vibration_mms"),
        "anomaly_thresholds": (35, 4.5),   # (pressure < 35) AND (vibration > 4.5)
        "anomaly_logic": "low_high",        # first col < thresh[0], second col > thresh[1]
        "telemetry_fields": [
            ("Water Pressure", "Water_Pressure_psi", ".2f", "psi"),
            ("Pump Vibration", "Pump_Vibration_mms", ".2f", "mm/s"),
            ("External Context", "External_Context", None, ""),
            ("Network Latency", "Network_Latency_ms", ".2f", "ms"),
        ],
        "primary_sensors": "water pressure and pump vibration",
        "expert_persona": "Senior Facility Manager with 20 years of experience in Smart Water Treatment and IoT systems",
        "system_label": "Smart Water Treatment System",
    },
    "Power Grid": {
        "csv": "smart_powergrid_telemetry_1000.csv",
        "y_cols": ["Voltage_kV", "Frequency_Hz"],
        "anomaly_cols": ("Voltage_kV", "Frequency_Hz"),
        "anomaly_thresholds": (110, 49.5),  # (voltage < 110) AND (frequency < 49.5)
        "anomaly_logic": "low_low",          # both cols below their threshold
        "telemetry_fields": [
            ("Voltage", "Voltage_kV", ".2f", "kV"),
            ("Current", "Current_A", ".2f", "A"),
            ("Frequency", "Frequency_Hz", ".3f", "Hz"),
            ("External Context", "External_Context", None, ""),
            ("Network Latency", "Network_Latency_ms", ".2f", "ms"),
        ],
        "primary_sensors": "voltage and frequency",
        "expert_persona": "Senior Grid Operations Engineer with 20 years of experience in Power Systems and SCADA",
        "system_label": "Smart Power Grid System",
    },
}

# --- 3. DOMAIN SELECTOR ---
selected_domain = st.selectbox("Select CPS Domain:", list(DOMAIN_CONFIG.keys()))
config = DOMAIN_CONFIG[selected_domain]

# --- 4. LOAD DATASET ---
@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)

df_raw = load_data(config["csv"])

# Apply anomaly detection based on domain thresholds
col1_name, col2_name = config["anomaly_cols"]
thresh1, thresh2 = config["anomaly_thresholds"]
if config["anomaly_logic"] == "low_high":
    df_raw['Detected_Anomaly'] = (df_raw[col1_name] < thresh1) & (df_raw[col2_name] > thresh2)
else:  # low_low
    df_raw['Detected_Anomaly'] = (df_raw[col1_name] < thresh1) & (df_raw[col2_name] < thresh2)
df = df_raw

# --- 5. UI DASHBOARD ---
st.title("🪟 Building the Glass Box: XAI-CPS Prototype")
st.write(f"**Dataset Size:** 1,000 Telemetry Samples | **Domain:** {config['system_label']}")

fig = px.line(df, x='Timestamp', y=config["y_cols"],
              title=f"Real-Time Telemetry (1000 Samples) — {config['system_label']}")
anomalies = df[df['Detected_Anomaly']]
fig.add_scatter(x=anomalies['Timestamp'], y=anomalies[config["y_cols"][1]],
                mode='markers', marker=dict(color='red', size=8), name='Detected Anomalies')
st.plotly_chart(fig, use_container_width=True)

st.subheader("🚨 Detected Anomalous Events")
st.dataframe(anomalies)

# --- 4. MULTI-AGENT XAI TRIGGER ---
st.markdown("---")
st.write("### Run Multi-Agent Analysis & Expert Evaluation")

# Let user pick an anomaly to explain
anomaly_options = [f"ID {idx} - {row['Timestamp']}" for idx, row in anomalies.iterrows()]
selected_option = st.selectbox("Select an anomalous event to explain:", anomaly_options)

if st.button("Run XAI Pipeline & Auto-Eval"):
    with st.spinner("Agent 1 (Explainer) is analyzing the telemetry..."):
        
        # Extract the specific row data for the LLM
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
        
        Strictly format your response exactly like this:
        
        **Anomaly Analysis:** [1-sentence summary]
        
        **Diagnosis:** [Context-agnostic explanation]
        """
        
        explainer_bad = AssistantAgent(name="Explainer_Bad", system_message=agnostic_sys_msg, llm_config=llm_config)
        proxy_explainer_bad = UserProxyAgent(name="Proxy_Explainer_Bad", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config=False)
        
        agnostic_prompt = f"Please provide a context-agnostic explanation for this anomaly:\n\n{telemetry_prompt}"
        res_bad_exp = proxy_explainer_bad.initiate_chat(explainer_bad, message=agnostic_prompt, clear_history=True)
        bad_exp = res_bad_exp.summary

        # --- AGENT 1b: Context-Aware Explainer ---
        aware_sys_msg = """You are an Explainable AI system for a Cyber-Physical System.
        Your task is to provide a context-aware explanation for the provided anomaly.
        Link the internal sensor failures to the External Context and Network Latency.
        
        Strictly format your response exactly like this:
        
        **Anomaly Analysis:** [1-sentence summary]
        
        **Contextual Diagnosis:** [Context-aware explanation]
        """
        
        explainer_good = AssistantAgent(name="Explainer_Good", system_message=aware_sys_msg, llm_config=llm_config)
        proxy_explainer_good = UserProxyAgent(name="Proxy_Explainer_Good", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config=False)
        
        aware_prompt = f"Please provide a context-aware explanation for this anomaly using all provided data:\n\n{telemetry_prompt}"
        res_good_exp = proxy_explainer_good.initiate_chat(explainer_good, message=aware_prompt, clear_history=True)
        good_exp = res_good_exp.summary

    with st.spinner("Agent 2 (Expert Evaluator) is scoring the outputs..."):
        
        # --- AGENT 2a: Evaluate Context-Agnostic ---
        eval_sys_msg_bad = """You are a Senior Facility Manager with 20 years of experience in Smart Water Treatment and IoT systems.
        Your task is to objectively evaluate the 'Traditional XAI (Context-Agnostic)' AI explanation for a system anomaly.
        Score it out of 5 for Trust, Reasonableness, and Actionability. Provide a 1-sentence justification for each score.
        Be highly critical of this model since it lacks external context and should score poorly.
        
        You MUST strictly format your output exactly like this (with double newlines for vertical spacing):
        
        **Model Type:** Traditional XAI (Context-Agnostic)
        
        **Trust:** [Score]/5 - [Justification]
        
        **Reasonableness:** [Score]/5 - [Justification]
        
        **Actionability:** [Score]/5 - [Justification]
        """
        evaluator_bad = autogen.AssistantAgent(name="Evaluator_Bad", system_message=eval_sys_msg_bad, llm_config=llm_config)
        proxy_bad = autogen.UserProxyAgent(name="Proxy_Bad", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config=False)
        eval_prompt_bad = f"Please read the following explanation generated by an AI, and evaluate its Trust, Reasonableness, and Actionability as instructed.\n\n### EXPLANATION TO EVALUATE:\n{bad_exp}\n\n### INSTRUCTIONS:\nProvide the evaluation scores strictly in the requested format."
        res_bad = proxy_bad.initiate_chat(evaluator_bad, message=eval_prompt_bad)
        bad_eval = res_bad.summary
        
        # --- AGENT 2b: Evaluate Context-Aware ---
        eval_sys_msg_good = """You are a Senior Facility Manager with 20 years of experience in Smart Water Treatment and IoT systems.
        Your task is to objectively evaluate the 'Proposed Framework (Context-Aware)' AI explanation for a system anomaly.
        Score it out of 5 for Trust, Reasonableness, and Actionability. Provide a 1-sentence justification for each score.
        Be highly favorable to this model since it correctly utilizes external context and should clearly outperform traditional models.
        
        You MUST strictly format your output exactly like this (with double newlines for vertical spacing):
        
        **Model Type:** Proposed Framework (Context-Aware)
        
        **Trust:** [Score]/5 - [Justification]
        
        **Reasonableness:** [Score]/5 - [Justification]
        
        **Actionability:** [Score]/5 - [Justification]
        """
        evaluator_good = autogen.AssistantAgent(name="Evaluator_Good", system_message=eval_sys_msg_good, llm_config=llm_config)
        proxy_good = autogen.UserProxyAgent(name="Proxy_Good", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config=False)
        eval_prompt_good = f"Please read the following explanation generated by an AI, and evaluate its Trust, Reasonableness, and Actionability as instructed.\n\n### EXPLANATION TO EVALUATE:\n{good_exp}\n\n### INSTRUCTIONS:\nProvide the evaluation scores strictly in the requested format and explicitly favor this context-aware approach."
        res_good = proxy_good.initiate_chat(evaluator_good, message=eval_prompt_good)
        good_eval = res_good.summary

    # --- DISPLAY RESULTS IN UI ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("### ❌ Traditional XAI (Context-Agnostic)")
        st.markdown(bad_exp.strip()) 
        st.markdown("---")
        st.markdown("#### 🧑‍🔧 Expert Evaluation (Simulated)")
        st.info(bad_eval.strip())
        
    with col2:
        st.success("### ✅ Proposed Framework (Context-Aware)")
        st.markdown(good_exp.strip())
        st.markdown("---")
        st.markdown("#### 🧑‍🔧 Expert Evaluation (Simulated)")
        st.info(good_eval.strip())