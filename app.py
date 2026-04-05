import streamlit as st
import os
import sys
import pandas as pd
from typing import Dict, Any

# Adjust path for internal imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.predict import predict_intent

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="IntentIQ: LeadIntent AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR CARDS ---
st.markdown("""
<style>
    .intent-card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
        font-size: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .high-intent { background-color: #28a745; }
    .medium-intent { background-color: #fd7e14; }
    .low-intent { background-color: #6c757d; }
    .inquiry { background-color: #007bff; }
    .complaint { background-color: #dc3545; }
    .uncertain { background-color: #f1f3f5; color: #495057; border: 1px solid #dee2e6; }
    
    .stProgress > div > div > div > div {
        background-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE FOR AUTO-FILL ---
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

def set_text(text: str):
    st.session_state['input_text'] = text

# --- HEADER SECTION ---
st.title("🧠 IntentIQ — AI-Powered Lead Intent Intelligence Engine")
st.markdown("""
<div style="margin-top: -15px; margin-bottom: 20px;">
    <strong>Classify, prioritize, and act on leads in real-time.</strong> &nbsp; 
    <span style="background-color: #e9ecef; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">⚡ Real-time AI Inference</span>
</div>
""", unsafe_allow_html=True)
st.divider()

# --- LAYOUT: 2 COLUMNS ---
col_input, col_output = st.columns([1, 1.2], gap="large")

# --- LEFT COLUMN: INPUT ---
with col_input:
    st.markdown("### 📝 Interaction Input")
    st.write("Enter the customer's message or sales note below to analyze intent.")
    
    # Text input with session state
    user_input = st.text_area(
        "Customer Message / Sales Note:",
        value=st.session_state['input_text'],
        height=150,
        placeholder="e.g., We need the pricing for the enterprise plan ASAP."
    )
    
    predict_clicked = st.button("🚀 Analyze Intent", use_container_width=True, type="primary")

    st.markdown("#### ✨ Quick Examples")
    st.write("Click a button below to quickly populate the input area.")
    
    # 2x2 grid for example buttons
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        if st.button("Immediate Buy", use_container_width=True):
            set_text("Need pricing and demo for the enterprise plan ASAP")
        if st.button("Bug Complaint", use_container_width=True):
            set_text("Your dashboard is crashing when I try to save leads")
    with e_col2:
        if st.button("Non-Committal", use_container_width=True):
            set_text("Interesting tool, but not sure yet, maybe next year")
        if st.button("Basic Inquiry", use_container_width=True):
            set_text("What are your security features?")

# --- RIGHT COLUMN: OUTPUT ---
with col_output:
    st.markdown("### 📊 Prediction Dashboard")
    
    if predict_clicked or (user_input and not st.session_state['input_text']):
        if not user_input.strip():
            st.error("Please enter some text before analyzing.")
        else:
            with st.spinner("🧠 Analyzing linguistic patterns..."):
                try:
                    result = predict_intent(user_input)
                    intent = result["intent"]
                    conf = result["confidence"]
                    probs = result["probabilities"]
                    
                    # 0. Business Action Mapping
                    suggested_actions = {
                        "High Intent": "Prioritize immediately (Hot lead)",
                        "Medium Intent": "Nurture and follow up",
                        "Low Intent": "Deprioritize or filter out",
                        "Inquiry": "Provide information / route to sales",
                        "Complaint": "Escalate to support immediately",
                        "Uncertain": "Needs more context"
                    }
                    action_text = suggested_actions.get(intent, "Awaiting context")

                    # 1. Styled Intent Display
                    style_class = intent.lower().replace(" ", "-")
                    icon_map = {
                        "High Intent": "🚀", "Medium Intent": "⚖️", "Low Intent": "📉",
                        "Inquiry": "🔍", "Complaint": "⚠️", "Uncertain": "🤔"
                    }
                    icon = icon_map.get(intent, "🧠")
                    
                    st.markdown(f"**Detected Buyer Intent**")
                    st.markdown(f"""
                    <div class="intent-card {style_class}">
                        {icon} {intent}
                    </div>
                    """, unsafe_allow_html=True)

                    # 1.1 Suggested Action Box (Inherits styling)
                    st.markdown(f"**💡 Suggested Action**")
                    st.markdown(f"""
                    <div class="intent-card {style_class}" style="font-size: 18px; padding: 15px; margin-top: -10px;">
                        {action_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # 2. Confidence Visuals
                    m_col1, m_col2 = st.columns([1, 2])
                    with m_col1:
                        st.metric("Prediction Confidence", f"{conf * 100:.2f}%")
                    with m_col2:
                        st.write("Certainty Index")
                        st.progress(conf)
                    
                    # 3. Interpretation Layer (Refined SaaS Thresholds)
                    st.markdown("#### 💡 Prediction Interpretation")
                    
                    if conf < 0.50:
                        st.info(f"⚠️ **Low Confidence** — The model detects linguistic ambiguity. High risk of misclassification.")
                        if intent == "Inquiry":
                            st.write("*Note: Model detects an informational query but lacks strong intent signals.*")
                    elif conf < 0.70:
                        st.warning(f"⚖️ **Moderate Confidence** — Prediction is likely valid for **{intent}**, but nuances overlap with other categories.")
                    else:
                        st.success(f"✅ **High Confidence** — The system is highly confident in this prioritization signal.")

                    # 4. Explainability (Top Signals)
                    st.markdown("#### 🔍 Why this prediction? (Top 2 Signals)")
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    for label, prob in sorted_probs[:2]:
                        st.markdown(f"- **{label}**: `{prob:.4f}`")
                        
                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")
                    st.info("Ensure you have trained the model using `python model/train.py` first.")
    else:
        st.info("Input a customer message on the left to see the AI prediction engine in action.")

# --- SIDEBAR & ABOUT MODEL ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png")
    st.markdown("### 🛡️ About IntentIQ")
    st.markdown("""
    **IntentIQ** is a production-grade NLP classifier designed for high-stakes sales triage. 
    
    It categorizes linguistic patterns into **5 core categories** to help sales teams prioritize their pipeline effectively.
    """)
    
    with st.expander("🛠️ Model Architecture"):
        st.write("**Model:** Linear SVM / LogReg (Calibrated)")
        st.write("**Vectorizer:** TF-IDF (Unigram + Bigram)")
        st.write("**Training Data:** 600+ Stylized Sales Notes")
        
        # Check if confusion matrix exists to show
        cm_path = os.path.join(os.path.dirname(__file__), "model", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix Visualization")
        else:
            st.caption("Confusion matrix not found. Run training to generate.")

# --- FOOTER ---
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    Built with Scikit-learn + FastAPI + Streamlit | Part of LeadIntent AI Pipeline
</div>
""", unsafe_allow_html=True)
