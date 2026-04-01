import streamlit as st
import pandas as pd
import plotly.express as px
import autogen

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Glass Box XAI", layout="wide")

llm_config = {
    "config_list": [{"model": "llama3.2", "base_url": "http://localhost:11434/v1", "api_key": "ollama"}],
    "temperature": 0.2
}

# --- 2. LOAD THE 1000-SAMPLE DATASET ---
@st.cache_data
def load_data():
    df = pd.read_csv('smart_water_telemetry_1000.csv')
    # Simple Anomaly Detection Logic (Z-Score/Thresholding)
    df['Detected_Anomaly'] = (df['Water_Pressure_psi'] < 35) & (df['Pump_Vibration_mms'] > 4.5)
    return df

df = load_data()

# --- 3. UI DASHBOARD ---
st.title("🪟 Building the Glass Box: XAI-CPS Prototype")
st.write("**Dataset Size:** 1,000 Telemetry Samples | **Domain:** Smart Water Treatment System")

# Plot the 1000 samples
fig = px.line(df, x='Timestamp', y=['Water_Pressure_psi', 'Pump_Vibration_mms'], 
              title="Real-Time Telemetry (1000 Samples)")
# Add red dots for detected anomalies
anomalies = df[df['Detected_Anomaly']]
fig.add_scatter(x=anomalies['Timestamp'], y=anomalies['Pump_Vibration_mms'], 
                mode='markers', marker=dict(color='red', size=8), name='Detected Anomalies')
st.plotly_chart(fig, use_container_width=True)

st.subheader("🚨 Detected Anomalous Events")
st.dataframe(anomalies.head(10))

# --- 4. MULTI-AGENT XAI TRIGGER ---
st.markdown("---")
st.write("### Run Multi-Agent Analysis & Expert Evaluation")

# Let user pick an anomaly to explain
anomaly_timestamps = anomalies['Timestamp'].tolist()
selected_time = st.selectbox("Select an anomalous timestamp to explain:", anomaly_timestamps)

if st.button("Run XAI Pipeline & Auto-Eval"):
    with st.spinner("Agent 1 (Explainer) is analyzing the telemetry..."):
        
        # Extract the specific row data for the LLM
        row_data = df[df['Timestamp'] == selected_time].iloc[0]
        telemetry_prompt = f"""
        Anomaly detected at {selected_time}.
        - Water Pressure: {row_data['Water_Pressure_psi']:.2f} psi
        - Pump Vibration: {row_data['Pump_Vibration_mms']:.2f} mm/s
        - Weather Context: {row_data['Weather_Context']}
        - Network Latency: {row_data['Network_Latency_ms']:.2f} ms
        """

        # --- AGENT 1: The Explainer ---
        xai_explainer = autogen.AssistantAgent(
            name="XAI_Explainer",
            system_message="""You are an Explainable AI system for a Cyber-Physical System. 
            Provide two explanations for the provided anomaly. 
            Strictly format your response exactly like this:
            
            **Anomaly Analysis:** [1-sentence summary]
            
            **Diagnosis:** [Context-agnostic explanation using ONLY internal pressure/vibration data. Assume mechanical failure.]
            
            ===SPLIT===
            
            **Anomaly Analysis:** [1-sentence summary]
            
            **Contextual Diagnosis:** [Context-aware explanation linking internal sensor failures to the External Weather/Latency Context]
            """,
            llm_config=llm_config,
        )
        
        user_proxy_1 = autogen.UserProxyAgent(
            name="User_Proxy_1",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False
        )

        chat_res_1 = user_proxy_1.initiate_chat(xai_explainer, message=telemetry_prompt)
        explanation_result = chat_res_1.summary

        if "===SPLIT===" in explanation_result:
            bad_exp, good_exp = explanation_result.split("===SPLIT===")
        else:
            bad_exp = explanation_result
            good_exp = "*Context-aware explanation generation failed.*"

    with st.spinner("Agent 2 (Expert Evaluator) is scoring the outputs..."):
        
        # --- AGENT 2: The LLM-as-a-Judge ---
        expert_evaluator = autogen.AssistantAgent(
            name="Expert_Evaluator",
            system_message="""You are a Senior Facility Manager with 20 years of experience in Smart Water Treatment and IoT systems.
            Your task is to objectively evaluate two AI explanations for a recent system anomaly.
            Score each out of 5 for Trust, Reasonableness, and Actionability. Provide a 1-sentence justification for each score.
            
            You MUST strictly format your output exactly like this to allow for UI parsing:
            
            **Trust:** [Score]/5 - [Justification]
            **Reasonableness:** [Score]/5 - [Justification]
            **Actionability:** [Score]/5 - [Justification]
            
            ===EVAL_SPLIT===
            
            **Trust:** [Score]/5 - [Justification]
            **Reasonableness:** [Score]/5 - [Justification]
            **Actionability:** [Score]/5 - [Justification]
            """,
            llm_config=llm_config,
        )

        user_proxy_2 = autogen.UserProxyAgent(
            name="User_Proxy_2",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False
        )

        eval_prompt = f"""
        Please evaluate these two explanations for the anomaly:
        
        EXPLANATION 1 (Context-Agnostic):
        {bad_exp}
        
        EXPLANATION 2 (Context-Aware):
        {good_exp}
        """

        chat_res_2 = user_proxy_2.initiate_chat(expert_evaluator, message=eval_prompt)
        eval_result = chat_res_2.summary

        if "===EVAL_SPLIT===" in eval_result:
            bad_eval, good_eval = eval_result.split("===EVAL_SPLIT===")
        else:
            bad_eval = eval_result
            good_eval = "*Evaluation parsing failed.*"

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