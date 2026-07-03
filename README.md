# 🪟 Building the Glass Box: A Human-Centered Framework for Explainable AI in Cyber-Physical Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)](https://ollama.ai/)
[![Microsoft AutoGen](https://img.shields.io/badge/Microsoft_AutoGen-Multi--Agent-0078D4)](https://microsoft.github.io/autogen/)

## 📖 Overview
The integration of Artificial Intelligence (AI) and Cyber-Physical Systems (CPS) is driving a new industrial transformation. However, the "black box" nature of high-performance AI models creates catastrophic risks in safety-critical systems. 

This repository contains the empirical implementation and research report for our B.Tech project. It addresses a critical failure in current Explainable AI (XAI) methods—a **lack of context-awareness**—by proposing and implementing a novel, human-centered methodological framework. 

We transition from theoretical design to a functional "Glass Box" software prototype. To prove scalability, the system is evaluated on robust, synthesized datasets of **1,000 time-series samples** across multiple domains (a **Smart Water Treatment System** and a **Smart Power Grid**), demonstrating how context-aware AI outperforms traditional context-agnostic models in real-world pipelines.

## ✨ Key Features
* **Multi-Domain Simulation & Hybrid Pipeline:** Replicates real-world industrial pipelines for both Smart Water and Power Grid systems by applying algorithmic anomaly detection across 1,000-sample datasets, followed by dynamic LLM-based explanations.
* **Advanced Multi-Agent Architecture:** Utilizes Microsoft AutoGen to orchestrate multiple AI agents—including specialized **XAI Explainers** and **Expert Evaluators** (LLM-as-a-judge)—to autonomously analyze sensor telemetry and score explanations.
* **100% Local & Secure Data Processing:** Addresses the privacy and cybersecurity vulnerabilities of cloud APIs by running **Llama 3.2** entirely offline via Ollama.
* **Context-Aware Explanations:** Correlates internal physical sensor deviations (e.g., pressure drops, voltage sags) with external environmental contexts (e.g., storms, extreme heat waves, network latency).
* **Batch Evaluation Pipeline:** Automatically processes *N* anomalies end-to-end (explain → judge → aggregate), producing per-agent average scores, per-metric winner breakdowns, and downloadable CSV reports—all driven by a single button click.
* **Premium Human-Centered Dashboard:** A completely revamped Streamlit interface featuring a dynamic domain selector, modern "Glass Box" aesthetics, side-by-side comparative analysis, structured evaluation scorecards, and the new batch evaluation panel.

## 📂 Repository Structure
* `/code/` - Contains the Python scripts:
  * `app.py`: The Streamlit dashboard, domain selector, AutoGen multi-agent system (Explainers and Evaluators), and the batch evaluation pipeline with aggregate scoring.
  * `generate_dataset.py`: Generates the 1,000-sample Smart Water telemetry dataset.
  * `generate_powergrid_dataset.py`: Generates the 1,000-sample Smart Power Grid telemetry dataset.
* `/reports/` - Contains the compiled research reports, journal paper PDF (`XAI_CPS_Journal.pdf`), and presentation materials.
* `/assets/` - Contains dashboard screenshots, output images, and architectural diagrams used in the evaluation phase.

## 📸 Dashboard & Outputs

<!-- USER INSTRUCTION: ADD YOUR DASHBOARD SCREENSHOTS BELOW -->

### 1. The "Glass Box" Interface & Domain Selector
![Streamlit Interface](assets/tele.png)
![Streamlit Interface](assets/anomaly.png)
![Streamlit Interface](assets/select.png)
*Real-time CPS sensor telemetry visualization with the ability to switch between Smart Water and Power Grid domains, dynamically highlighting detected anomalies.*

### 2. Multi-Agent Analysis & Structured Explanations
![XAI Explanations](assets/o1.png)
*The multi-agent system generates structured, bulleted diagnoses. The Context-Aware model successfully links internal failures (e.g., voltage sags) to external events, unlike the Context-Agnostic model.*

### 3. Automated Expert Evaluation & Comparison
![Expert Evaluation](assets/o2.png)
![Expert Evaluation](assets/table.png)
*An Expert Evaluator agent automatically scores both explanations out of 5 across Trust, Reasonableness, and Actionability, summarized in a Quick Comparison Table.*

### 4. Batch Evaluation — Aggregate Scores Across Anomalies
![Batch Evaluation Slider](assets/batch-1.png)
*Select the number of anomalies to process (up to all detected events) and launch batch evaluation with a single click.*

![Batch Summary Table](assets/Batch-2.png)
*Aggregate batch summary showing average Trust, Reasonableness, and Actionability scores per agent, with the Δ (improvement) column and overall winner declaration.*

![Per-Metric Breakdown](assets/Batch-3.png)
*Per-metric winner breakdown table with expandable raw per-anomaly scores and a CSV download button for further analysis.*

![Per-Anomaly Raw Scores](assets/Batch-4.png)
*Expandable view of raw per-anomaly scores across both agents, showing Trust, Reasonableness, and Actionability for every processed event.*

## 🚀 Getting Started (Running Locally)

### Prerequisites
1. Install [Python 3.9+](https://www.python.org/downloads/).
2. Install [Ollama](https://ollama.com/) and download the Llama 3.2 model:
   
   ```bash
   ollama run llama3.2
   ```
### Installation
1. Clone the repository:
   
   ```bash
   git clone https://github.com/yourusername/glass-box-xai-cps.git
   cd glass-box-xai-cps
   ```
   
2. Install the required dependencies:
   
   ```bash
   pip install streamlit pandas plotly ag2[openai]
   ```

### Execution

1. **Generate The Datasets** : First synthesize the 1000-sample telemetry datasets by running the generation scripts:
   ```bash
   python code/generate_dataset.py
   python code/generate_powergrid_dataset.py
   ```
2. Ensure the Ollama application is running in the background.
3. Launch the Streamlit dashboard:
   
   ```bash
   streamlit run code/app.py
   ```
   
4. Open the provided local URL (usually ```http://localhost:8501```) in your browser.
5. **Single-Event Analysis:** Select an anomalous event from the dropdown and click **"🚀 Run XAI Pipeline & Auto-Eval"** to trigger the local Llama 3.2 multi-agent analysis for a single anomaly.
6. **Batch Evaluation:** Scroll to the **"🧪 Batch Evaluation"** section, choose the number of anomalies with the slider, and click **"🧪 Run Batch Eval"** to process multiple anomalies end-to-end and view aggregate scores.

## 📊 Phase 2: Multi-Agent Automated Evaluation
As part of this framework, we have upgraded our manual human evaluation process into a sophisticated **LLM-as-a-Judge** pipeline. A designated **Expert Evaluator Agent** (configured with domain-specific personas like Senior Grid Operations Engineer) automatically grades the contrasting explanations in real-time.

The evaluation utilizes a 5-point scale based on:
1. **Reasonableness**
2. **Trust**
3. **Actionability**

The dashboard instantly displays these scorecards and justifications alongside a Quick Comparison Table, demonstrating that context-aware explanations consistently achieve higher trust and actionability scores.

## 🧪 Phase 3: Batch Evaluation & Aggregate Scoring
To move beyond single-sample analysis and provide statistically meaningful results, we introduced a **Batch Evaluation** pipeline that processes multiple anomalies autonomously:

| Feature | Description |
|---|---|
| **Configurable Sample Size** | A slider lets you choose how many anomalies (1 → all detected) to evaluate |
| **End-to-End Pipeline** | For each anomaly: Explain (Agnostic) → Explain (Aware) → Judge (Agnostic) → Judge (Aware) |
| **Aggregate Summary Table** | Average Trust, Reasonableness, and Actionability per agent with a Δ (Aware − Agnostic) column |
| **Overall Winner** | Automatically declares the winning agent based on the mean of all three metrics |
| **Per-Metric Breakdown** | Shows which agent wins on each individual metric |
| **Raw Score Inspector** | Expandable per-anomaly scores for full transparency |
| **CSV Export** | One-click download of all batch scores for offline analysis or inclusion in research papers |

This batch pipeline enables reproducible, multi-sample empirical validation — a key requirement for publishing the framework's results in academic venues.

## 🏗️ Architecture

The system employs a modular multi-agent architecture with reusable helper functions:

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                       │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Domain       │  │ Single-Event │  │ Batch Evaluation  │ │
│  │ Selector     │  │ Analysis     │  │ Pipeline          │ │
│  └──────────────┘  └──────┬───────┘  └────────┬──────────┘ │
│                           │                    │            │
│              ┌────────────┴────────────────────┘            │
│              ▼                                              │
│  ┌───────────────────────────────────────────────┐          │
│  │         Reusable Agent Helper Functions        │          │
│  │  build_telemetry_prompt()                      │          │
│  │  run_explainer_agnostic() / _aware()           │          │
│  │  run_evaluator_agnostic() / _aware()           │          │
│  │  parse_scores()                                │          │
│  └────────────────────┬──────────────────────────┘          │
│                       ▼                                     │
│  ┌─────────────────────────────────────────────┐            │
│  │    Microsoft AutoGen Multi-Agent System      │            │
│  │  ┌──────────────┐  ┌────────────────────┐   │            │
│  │  │ Explainer    │  │ Expert Evaluator    │   │            │
│  │  │ Agents (×2)  │  │ Agents (×2)        │   │            │
│  │  └──────┬───────┘  └────────┬───────────┘   │            │
│  │         └───────────┬───────┘               │            │
│  │                     ▼                       │            │
│  │        Llama 3.2 via Ollama (Local)         │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## 👥 Authors & Acknowledgements
### Researchers
* Subhranshu Panda (Dept. of Computer Science Engineering, IIIT Bhubaneswar)
* Shreyansh Gupta (Dept. of Computer Science Engineering, IIIT Bhubaneswar)

### Project Guide:
* Prof. Bharati Mishra (IIIT Bhubaneswar)

This project was completed in partial fulfillment of the requirements for the degree of Bachelor of Technology in Computer Science Engineering.

---

## 📝 License
This project is for academic and research purposes.
