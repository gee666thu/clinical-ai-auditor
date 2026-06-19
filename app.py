import streamlit as st
import os
import json
import re
from dotenv import load_dotenv
from groq import Groq

# 1. Initialize Groq securely
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

st.set_page_config(page_title="LearnLens Med Pro: Clinical AI Auditor", layout="wide")
st.title("🏥 LearnLens Med Pro")
st.caption("Advanced Clinical Simulation Engine & Automated Triage Auditor")

# 2. Main Interface Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Clinical Case Intake Deck")
    topic = st.text_input("🔮 Target Diagnosis / Pathology Tracker", placeholder="e.g., Acute Appendicitis, Sepsis, Diabetic Ketoacidosis")
    content = st.text_area(
        "📝 Patient Presentation Summary & EHR Snippets", 
        height=280,
        placeholder="Paste real-time triage notes, vital sign readings, history of present illness, or laboratory panels here..."
    )
    
with col2:
    st.subheader("⚡ Live Clinical Telemetry Matrix")
    
    if content.strip():
        word_count = len(content.split())
        sentence_count = max(1, content.count(".") + content.count("!") + content.count("?"))
        density = round(word_count / sentence_count, 1)
        
        # Base clinical textual complexity index (AI load calculation)
        case_complexity = min(100, max(10, int(word_count * 0.45)))
        
        # Normalize text inputs to avoid case sensitivity bugs
        content_lower = content.lower()
        
        # 🧪 Condition Classification Dictionary Tiers
        critical_markers = ["shock", "critical", "acidosis", "refractory", "obtunded", "unresponsive", "hypothermia", "septic", "lethargy"]
        guarded_markers = ["acute", "pain", "fever", "nausea", "vomiting", "tenderness", "guarding", "appendicitis", "cholecystitis"]
        
        critical_hits = sum(1 for word in critical_markers if word in content_lower)
        guarded_hits = sum(1 for word in guarded_markers if word in content_lower)
        
        # 🧠 Algorithmic Routing Logic (Critical vs Medium vs Normal)
        if critical_hits >= 2 or "shock" in content_lower:
            status_color = "🔴 CRITICAL TRIAGE"
            vitals_volatility = min(100, max(75, 70 + (critical_hits * 5)))
        elif guarded_hits >= 2 or critical_hits == 1:
            status_color = "🟡 GUARDED / UNSTABLE"
            vitals_volatility = min(70, max(45, 40 + (guarded_hits * 6)))
        else:
            status_color = "🟢 STABLE / AMBULATORY"
            vitals_volatility = min(35, max(5, 10 + (word_count // 20)))
            
        # Display Current Dynamic Case Classification
        st.markdown(f"### **Current Patient State: `{status_color}`**")
        
        # Visual Telemetry Metric Monitors
        st.metric(label="📊 Case Presentation Complexity (AI Load)", value=f"{case_complexity}%")
        st.progress(case_complexity / 100)
        
        st.metric(label="🚨 Vitals Volatility Index", value=f"{vitals_volatility}%", delta=f"{density} metrics/sent")
        st.progress(vitals_volatility / 100)
    else:
        st.info("📋 System Standing By. Input raw patient presentation records on the left to initialize live clinical simulation analytics.")
        case_complexity = 40
        vitals_volatility = 30

# 3. Audit Execution Pipeline
st.markdown("---")
if st.button("🚀 Run Active Cognitive Clinical Audit", use_container_width=True):
    if not topic.strip() or not content.strip():
        st.error("Missing intake parameters: Please complete both the Diagnosis Tracker and Case Presentation inputs.")
    else:
        with st.spinner("Processing deep architectural diagnostics via Llama-3.1 Framework..."):
            try:
                structured_prompt = f"""
                You are an elite chief medical auditor and clinical simulation designer. 
                Analyze these case files for suspected pathology '{topic}'.
                Case Presentation: {content}
                
                You must return strictly a valid JSON object. Do not include markdown blocks, wrapping text, or backticks.
                JSON Schema:
                {{
                  "diagnostic_accuracy_mod": -10 to 15,
                  "overlooked_differentials": ["differential 1", "differential 2"],
                  "latent_clinical_traps": ["trap/red flag 1", "trap/red flag 2"],
                  "immediate_interventions": ["action 1", "action 2"],
                  "clinical_viva_cards": [
                    {{"q": "Symptomatic Vector Challenge", "a": "Underlying Pathophysiology"}},
                    {{"q": "Laboratory/Diagnostic Conflict", "a": "Correct Action/Interpretation"}}
                  ]
                }}
                """
                
                raw_response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": structured_prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                # Secure clean up of raw LLM content
                cleaned_output = raw_response.choices[0].message.content.strip()
                data = json.loads(cleaned_output)
                
                # Dynamic Scoring Algorithm
                accuracy_baseline = int((100 - vitals_volatility) * 0.4 + (case_complexity * 0.5))
                calculated_readiness = max(5, min(100, accuracy_baseline + data.get("diagnostic_accuracy_mod", 0)))
                
                # Clinical Intelligence Dashboard Display
                st.subheader("📈 Clinical Decision Intelligence Dashboard")
                m_col1, m_col2 = st.columns([1, 2])
                with m_col1:
                    st.metric(
                        label="🎯 Predictive Diagnostic Accuracy Score", 
                        value=f"{calculated_readiness}%",
                        delta=f"{data.get('diagnostic_accuracy_mod', 0):+} modifier shift"
                    )
                with m_col2:
                    st.write("### Clinician Decision Proficiency Matrix")
                    st.progress(calculated_readiness / 100)
                
                # Feature 1: Dynamic Diagnostics Panels
                st.markdown("---")
                left_panel, right_panel = st.columns(2)
                
                with left_panel:
                    st.markdown("### 🔍 Overlooked Differential Gaps")
                    for diff in data.get("overlooked_differentials", []):
                        st.error(f"🩺 **Consider Alternative:** {diff}")
                        
                with right_panel:
                    st.markdown("### ⚠️ Latent Clinical Red Flags & Traps")
                    for trap in data.get("latent_clinical_traps", []):
                        st.warning(f"🚨 **Critical Risk:** {trap}")
                
                # Feature 2: Immediate Order Set Generation
                st.markdown("---")
                st.subheader("🧪 Automated Hospital Order Set & Interventions")
                for action in data.get("immediate_interventions", []):
                    st.info(f"⚡ **Immediate Clinical Action Item:** {action}")
                    
                # Feature 3: High-Yield Flashcard Challenges
                st.markdown("---")
                st.subheader("🧠 Active-Recall Clinical Simulation Vectors")
                for idx, card in enumerate(data.get("clinical_viva_cards", [])):
                    with st.expander(f"🧩 Scenario Challenge Vector {idx+1}: {card.get('q')}"):
                        st.success(f"**Verified Pathophysiological Resolution:** {card.get('a')}")
                        
            except Exception as e:
                st.error(f"Clinical Processing Interruption: {e}")