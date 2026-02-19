import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import autogen
import os

# --- 1. MOCK DATA GENERATION ---
# Simulating a Smart Water Treatment System
def generate_mock_data():
    np.random.seed(42)
    time_steps = pd.date_range("2026-02-19 08:00", periods=50, freq="10min")
    
    # Normal operations
    pressure = np.random.normal(50, 2, 50)
    vibration = np.random.normal(2, 0.5, 50)
    context = ["Clear"] * 50
    
    # Inject anomaly at step 40 (Storm hits, pump overworks to compensate)
    pressure[40:45] = np.random.normal(30, 5, 5) # Sudden drop in pressure
    vibration[40:45] = np.random.normal(6, 1, 5) # Spike in vibration
    for i in range(40, 50):
        context[i] = "Heavy Storm & Network Latency"
        
    df = pd.DataFrame({
        "Time": time_steps,
        "Water_Pressure_psi": pressure,
        "Pump_Vibration_mm_s": vibration,
        "External_Context": context
    })
    return df

# --- 2. AUTOGEN MULTI-AGENT SETUP ---
def run_xai_analysis(anomaly_data):
    # Configure AutoGen to use your local offline Ollama model
    llm_config = {
        "config_list": [
            {
                "model": "llama3.2", 
                "base_url": "http://127.0.0.1:11434/v1", # Must be an underscore _
                "api_key": "ollama", # A dummy string is safer than 'NotRequired'
                "api_type": "openai" # Explicitly tell AutoGen to use the OpenAI local format
            }
        ],
        "temperature": 0.2, 
    }
    
    # Agent 1: The CPS Monitor (Detects the issue)
    cps_monitor = autogen.AssistantAgent(
        name="CPS_Monitor",
        system_message="You are an IoT sensor monitor. Review the data and state strictly what internal sensors failed. Do not look at external context.",
        llm_config=llm_config,
    )
    
    # Agent 2: The XAI Explainer (Applies your framework)
    xai_explainer = autogen.AssistantAgent(
        name="XAI_Explainer",
        system_message="""You are an Explainable AI system for a Cyber-Physical System. 
        You must provide two explanations for the anomaly. 
        
        Strictly format your response exactly like this:
        
        **Anomaly Analysis:**
        [Brief 1-sentence summary of the sensor data]
        
        **Diagnosis:**
        [Context-agnostic explanation using ONLY internal sensor data]
        
        ===SPLIT===
        
        **Anomaly Analysis:**
        [Brief 1-sentence summary of the sensor data]
        
        **Contextual Diagnosis:**
        [Context-aware explanation linking internal sensor failures to the External Context]
        """,
        llm_config=llm_config,
    )
    
    # Initiate chat (Simulating the backend process)
    chat_history = cps_monitor.initiate_chat(
        xai_explainer,
        message=f"Analyze this recent anomaly data and provide the explanations: {anomaly_data}",
        max_turns=1
    )
    
    return chat_history.chat_history[-1]['content']

# --- 3. STREAMLIT INTERFACE ---
st.set_page_config(layout="wide", page_title="XAI-CPS Glass Box")

st.title("🪟 Building the Glass Box: XAI-CPS Prototype")
st.markdown("Evaluating Phase 3 Context-Awareness in a Smart Water Treatment System")

df = generate_mock_data()

# Visualization
st.subheader("Real-Time CPS Sensor Telemetry")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Time'], y=df['Water_Pressure_psi'], mode='lines', name='Water Pressure (psi)'))
fig.add_trace(go.Scatter(x=df['Time'], y=df['Pump_Vibration_mm_s'], mode='lines', name='Pump Vibration (mm/s)', yaxis='y2'))

fig.update_layout(
    yaxis=dict(title="Pressure"),
    yaxis2=dict(title="Vibration", overlaying='y', side='right'),
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)



# Trigger the Agents
if st.button("🚨 Run Anomaly Detection & XAI Analysis"):
    with st.spinner("Multi-Agent System Analyzing Data..."):
        anomaly_window = df.iloc[38:45].to_dict() # Pass the window where the anomaly occurs
        explanation_result = run_xai_analysis(anomaly_window)
        
        st.divider()
        st.subheader("AI System Explanations")
        
        # Display the results side-by-side
        col1, col2 = st.columns(2)
        
        # Parse the LLM output using the delimiter we forced it to use
        if "===SPLIT===" in explanation_result:
            bad_explanation, good_explanation = explanation_result.split("===SPLIT===")
        else:
            # Fallback just in case the LLM disobeys the prompt format
            bad_explanation = explanation_result
            good_explanation = "*Context-aware explanation generating... please run again.*"
        
        with col1:
            st.error("### ❌ Traditional XAI (Context-Agnostic)")
            st.write("*(Simulating LIME/SHAP internal feature weighting)*")
            st.markdown(bad_explanation.strip()) 
            
        with col2:
            st.success("### ✅ Proposed Framework (Context-Aware)")
            st.write("*(Integrating physical sensor logic with external environment variables)*")
            st.markdown(good_explanation.strip())
            
        st.info("**Evaluation Phase 2 Next Step:** Present these two panels to domain experts and have them rate Actionability, Reasonableness, and Trust.")