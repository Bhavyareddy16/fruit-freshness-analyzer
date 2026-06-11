import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import cv2
from io import BytesIO
from datetime import datetime

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Import simulation utilities
from simulator import generate_telemetry_history, simulate_hsi_scan, WAVELENGTHS

# Page Config
st.set_page_config(
    page_title="Fruit Freshness Analyzer",
    page_icon="🍋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styles Injection
st.markdown("""
    <style>
    /* Dark Theme Core Adjustments */
    .stApp {
        background-color: #0E1117;
        color: #E2E8F0;
    }
    
    /* Modern Card Layout */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(148, 163, 184, 0.2);
    }
    
    /* Freshness Indicators */
    .fresh-badge {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .moderate-badge {
        background-color: rgba(234, 179, 8, 0.15);
        color: #FACC15;
        border: 1px solid rgba(234, 179, 8, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .spoiled-badge {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Headers Styling */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    .gradient-text {
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# Helper: Load Models with Auto-Training Check
@st.cache_resource
def load_ml_models():
    models = {}
    required_models = [
        "models/spectral_classifier.joblib",
        "models/enose_classifier.joblib",
        "models/sugar_brix_regressor.joblib",
        "models/moisture_pct_regressor.joblib",
        "models/firmness_n_regressor.joblib",
        "models/spoilage_regressor.joblib",
        "models/fruit_encoder.joblib",
        "models/freshness_encoder.joblib",
        "models/enose_scaler.joblib"
    ]
    
    # Check if models exist
    missing = [m for m in required_models if not os.path.exists(m)]
    if missing:
        return None
        
    try:
        models["spectral_classifier"] = joblib.load("models/spectral_classifier.joblib")
        models["enose_classifier"] = joblib.load("models/enose_classifier.joblib")
        models["sugar_regressor"] = joblib.load("models/sugar_brix_regressor.joblib")
        models["moisture_regressor"] = joblib.load("models/moisture_pct_regressor.joblib")
        models["firmness_regressor"] = joblib.load("models/firmness_n_regressor.joblib")
        models["spoilage_regressor"] = joblib.load("models/spoilage_regressor.joblib")
        models["fruit_encoder"] = joblib.load("models/fruit_encoder.joblib")
        models["freshness_encoder"] = joblib.load("models/freshness_encoder.joblib")
        models["enose_scaler"] = joblib.load("models/enose_scaler.joblib")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None
        
    return models

# UI Setup: Header
st.markdown("<h1 class='gradient-text'>Non-Destructive Fruit Freshness Analyzer</h1>", unsafe_allow_html=True)
st.markdown("##### Powered by Simulated Hyperspectral Imaging, IoT Electronic Nose, and Machine Learning Fusion")
st.markdown("---")

# Sidebar Status Panel
st.sidebar.markdown("### 🛠️ System Status")
models = load_ml_models()

if models is None:
    st.sidebar.warning("⚠️ Machine learning models not found.")
    st.sidebar.info("You need to generate data and train the models before analyzing fruits.")
    if st.sidebar.button("⚙️ Setup & Train Models Now", use_container_width=True):
        with st.spinner("Generating synthetic data..."):
            import subprocess
            subprocess.run([sys.executable, "generate_data.py"])
        with st.spinner("Training ML classifiers and regressors..."):
            subprocess.run([sys.executable, "train_models.py"])
            subprocess.run([sys.executable, "create_samples.py"])
        st.cache_resource.clear()
        st.rerun()
else:
    st.sidebar.success("✅ ML Pipeline: Online")
    st.sidebar.success("✅ Virtual Sensors: Calibrated")
    st.sidebar.info("Simulation Mode: Active (No hardware required)")

# Tabs Configuration
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Cold Storage IoT Dashboard", 
    "🍋 Fruit Freshness Scanner", 
    "📊 ML Evaluation Lab", 
    "📄 Quality Reports"
])

# ==========================================
# TAB 1: COLD STORAGE IoT DASHBOARD
# ==========================================
with tab1:
    st.markdown("### 🏢 Cold Storage Facility Telemetry")
    st.markdown("Real-time environmental tracking and spoilage risk monitoring inside fruit cold storage rooms.")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("##### Storage Controls")
        anomalous_mode = st.toggle("🚨 Simulate Ventilation Failure", value=False, 
                                   help="Simulates cooling unit breakdown leading to temperature spike and CO2 accumulation.")
        
        # Generate and cache telemetry history
        hist_df = generate_telemetry_history(hours=24, anomalous=anomalous_mode)
        
        current_temp = hist_df.iloc[-1]['temp_c']
        current_hum = hist_df.iloc[-1]['humidity_pct']
        current_co2 = hist_df.iloc[-1]['mq135_co2_ppm']
        current_alc = hist_df.iloc[-1]['mq3_alcohol_ppm']
        current_eth = hist_df.iloc[-1]['mq138_benzene_ppm']
        
        # Predict current spoilage index
        if models is not None:
            # Prepare feature row for current time
            # Assume Apple storage for simplicity in general dashboard
            fruit_enc = models["fruit_encoder"].transform(["Apple"])[0]
            feature_row = pd.DataFrame([[
                fruit_enc, current_temp, current_hum,
                current_co2, current_alc, 10.0, 2.0, current_eth
            ]], columns=['fruit_type_encoded', 'temp_c', 'humidity_pct', 
                         'mq135_co2_ppm', 'mq3_alcohol_ppm', 'mq4_methane_ppm', 
                         'mq137_ammonia_ppm', 'mq138_benzene_ppm'])
            
            spoilage_risk = models["spoilage_regressor"].predict(feature_row)[0]
        else:
            spoilage_risk = 0.15
            
        st.markdown("---")
        
        # Metrics Display
        st.markdown(f"""
            <div class='metric-card'>
                <p style='margin-bottom:0.25rem; font-size:0.9rem; color:#94A3B8;'>Spoilage Risk Index</p>
                <h2 style='margin:0; color:{"#F87171" if spoilage_risk > 0.6 else "#FACC15" if spoilage_risk > 0.3 else "#4ADE80"};'>{spoilage_risk*100:.1f}%</h2>
            </div>
            <br>
            <div class='metric-card'>
                <p style='margin-bottom:0.25rem; font-size:0.9rem; color:#94A3B8;'>Storage Temp</p>
                <h2 style='margin:0; color:{"#F87171" if current_temp > 22 else "#4ADE80"};'>{current_temp}°C</h2>
            </div>
            <br>
            <div class='metric-card'>
                <p style='margin-bottom:0.25rem; font-size:0.9rem; color:#94A3B8;'>Humidity</p>
                <h2 style='margin:0; color:#38BDF8;'>{current_hum}%</h2>
            </div>
        """, unsafe_allow_html=True)
        
        if anomalous_mode:
            st.error("🚨 CRITICAL ALERT: Cold storage temperature exceeds critical limits. Spoilage risk accelerating.")
        elif spoilage_risk > 0.3:
            st.warning("⚠️ VENTILATION WARNING: Minor gas accumulation detected. Recommended to flush room ventilation.")
            
    with col2:
        # Plotly Telemetry Graph
        fig = go.Figure()
        
        # Add primary axis lines (Temp & Humidity)
        fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['temp_c'], name="Temp (°C)",
                                 line=dict(color='#F87171', width=3)))
        fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['humidity_pct'], name="Humidity (%)",
                                 line=dict(color='#38BDF8', width=3), yaxis="y2"))
        
        # Add secondary axis lines (Gases)
        fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['mq135_co2_ppm'], name="CO₂ (ppm)",
                                 line=dict(color='#A855F7', width=2, dash='dot'), yaxis="y3"))
        fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['mq3_alcohol_ppm'], name="Ethanol (ppm)",
                                 line=dict(color='#F59E0B', width=2, dash='dot'), yaxis="y3"))
        fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['mq138_benzene_ppm'], name="Ethylene (ppm)",
                                 line=dict(color='#10B981', width=2, dash='dot'), yaxis="y3"))
        
        # Adjust Layout
        fig.update_layout(
            title="Cold Storage 24h Sensor Telemetry Timeline",
            xaxis=dict(title="Time", domain=[0.1, 0.9]),
            yaxis=dict(title=dict(text="Temperature (°C)", font=dict(color="#F87171")), tickfont=dict(color="#F87171")),
            yaxis2=dict(title=dict(text="Humidity (%)", font=dict(color="#38BDF8")), tickfont=dict(color="#38BDF8"),
                        anchor="free", overlaying="y", side="right", position=0.9),
            yaxis3=dict(title=dict(text="Gas Concentration (ppm)", font=dict(color="#A855F7")), tickfont=dict(color="#A855F7"),
                        anchor="free", overlaying="y", side="left", position=0.03),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark",
            height=500,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: FRUIT FRESHNESS SCANNER
# ==========================================
with tab2:
    st.markdown("### 🍋 Non-Destructive Scanning & Prediction")
    st.markdown("Upload a photo or select a pre-calibrated sample to perform simulated hyperspectral and E-nose sensing.")
    
    if models is None:
        st.info("💡 Please complete Model Setup in the sidebar first to access the scanner.")
    else:
        # Layout splits into setup and results
        scan_col1, scan_col2 = st.columns([2, 3])
        
        with scan_col1:
            st.markdown("##### 1. Select Fruit Profile")
            fruit_selection = st.selectbox("Fruit Category", ["Apple", "Banana", "Orange"])
            
            input_source = st.radio("Image Source", ["Preloaded Sample Gallery", "Upload Custom Image"])
            
            selected_img_path = None
            uploaded_image = None
            
            if input_source == "Preloaded Sample Gallery":
                # Create gallery columns
                gal1, gal2 = st.columns(2)
                
                # Check if sample files exist, generate if missing
                if not os.path.exists("sample_images/apple_fresh.png"):
                    import subprocess
                    subprocess.run([sys.executable, "create_samples.py"])
                    
                if fruit_selection == "Apple":
                    img_fresh = "sample_images/apple_fresh.png"
                    img_spoiled = "sample_images/apple_spoiled.png"
                elif fruit_selection == "Banana":
                    img_fresh = "sample_images/banana_fresh.png"
                    img_spoiled = "sample_images/banana_spoiled.png"
                else:
                    img_fresh = "sample_images/orange_fresh.png"
                    img_spoiled = "sample_images/orange_spoiled.png"
                    
                with gal1:
                    st.image(img_fresh, caption="Fresh Specimen", use_container_width=True)
                    if st.button("Select Fresh Sample", use_container_width=True):
                        st.session_state["active_sample"] = "fresh"
                        st.session_state["days_val"] = 1
                with gal2:
                    st.image(img_spoiled, caption="Spoiled Specimen", use_container_width=True)
                    if st.button("Select Spoiled Sample", use_container_width=True):
                        st.session_state["active_sample"] = "spoiled"
                        st.session_state["days_val"] = 12
                        
                # Default selection state handling
                active_sample = st.session_state.get("active_sample", "fresh")
                selected_img_path = img_fresh if active_sample == "fresh" else img_spoiled
                days_stored_default = st.session_state.get("days_val", 1)
                
            else:
                uploaded_image = st.file_uploader("Upload Fruit Photo (PNG, JPG)", type=["png", "jpg", "jpeg"])
                days_stored_default = 3
            
            st.markdown("##### 2. Configure Simulation Parameters")
            sim_days = st.slider("Storage Duration (Days)", min_value=1, max_value=20, value=days_stored_default, step=1)
            sim_temp = st.slider("Storage Temperature (°C)", min_value=4, max_value=35, value=20, step=1)
            
            st.markdown("---")
            run_scan = st.button("🚀 Execute Non-Destructive Scan", type="primary", use_container_width=True)
            
        with scan_col2:
            st.markdown("##### 3. Analysis Output & Virtual Sensors")
            
            if run_scan:
                # Progress Animations simulating scanning
                progress_container = st.empty()
                with progress_container.container():
                    st.info("📡 Scanning target specimen...")
                    bar = st.progress(0)
                    for percent in [25, 55, 80, 100]:
                        time.sleep(0.3)
                        bar.progress(percent)
                        if percent == 25:
                            st.info("🔦 Projecting light bands (Hyperspectral 400-1000nm scan)...")
                        elif percent == 55:
                            st.info("👃 Inhaling volatiles (E-Nose gas sweep)...")
                        elif percent == 80:
                            st.info("🤖 Fusing spectral-chemical data and ML predictions...")
                    st.success("🎉 Scan Complete!")
                    time.sleep(0.4)
                progress_container.empty()
                
                # Fetch simulated scan inputs
                scan_data = simulate_hsi_scan(fruit_selection, sim_days, sim_temp)
                
                # Prepare visual variables
                if uploaded_image is not None:
                    # Read image
                    img_bytes = uploaded_image.read()
                    pil_img = Image.open(BytesIO(img_bytes))
                    # Resize for standard
                    pil_img = pil_img.resize((400, 400))
                    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    img_cv = cv2.imread(selected_img_path)
                    
                # Run ML Prediction
                # A. HSI/NIR Prediction
                fruit_encoded = models["fruit_encoder"].transform([fruit_selection])[0]
                hsi_features = [fruit_encoded] + scan_data["spectrum"]
                hsi_df = pd.DataFrame([hsi_features], columns=['fruit_type_encoded'] + [f"band_{i+1}" for i in range(100)])
                
                # Classify
                hsi_pred_encoded = models["spectral_classifier"].predict(hsi_df)[0]
                hsi_probs = models["spectral_classifier"].predict_proba(hsi_df)[0]
                hsi_freshness = models["freshness_encoder"].inverse_transform([hsi_pred_encoded])[0]
                
                # Regress chemical traits
                pred_brix = models["sugar_regressor"].predict(hsi_df)[0]
                pred_moisture = models["moisture_regressor"].predict(hsi_df)[0]
                pred_firmness = models["firmness_regressor"].predict(hsi_df)[0]
                
                # B. E-Nose Prediction
                enose_features = pd.DataFrame([[
                    fruit_encoded, sim_temp, 75.0, # Assumed storage relative humidity
                    scan_data["gases"]["mq135"], scan_data["gases"]["mq3"],
                    scan_data["gases"]["mq4"], scan_data["gases"]["mq137"],
                    scan_data["gases"]["mq138"]
                ]], columns=['fruit_type_encoded', 'temp_c', 'humidity_pct', 
                             'mq135_co2_ppm', 'mq3_alcohol_ppm', 'mq4_methane_ppm', 
                             'mq137_ammonia_ppm', 'mq138_benzene_ppm'])
                
                scaled_enose = models["enose_scaler"].transform(enose_features)
                enose_pred_encoded = models["enose_classifier"].predict(enose_features)[0] # Gradient Boosting doesn't need scaling but uses same features
                enose_probs = models["enose_classifier"].predict_proba(enose_features)[0]
                enose_freshness = models["freshness_encoder"].inverse_transform([enose_pred_encoded])[0]
                
                # Ensembled freshness prediction
                combined_probs = (hsi_probs + enose_probs) / 2.0
                final_pred_encoded = np.argmax(combined_probs)
                final_freshness = models["freshness_encoder"].classes_[final_pred_encoded]
                final_confidence = combined_probs[final_pred_encoded]
                
                # Overlay computer vision markings (Spoilage warning or fresh checkmark)
                cv_img = img_cv.copy()
                h_img, w_img, _ = cv_img.shape
                if final_freshness == "Spoiled":
                    # Draw defective/soft rot bounding boxes as visual CV simulator
                    cv2.rectangle(cv_img, (int(w_img*0.25), int(h_img*0.25)), (int(w_img*0.75), int(h_img*0.75)), (0, 0, 220), 4)
                    cv2.putText(cv_img, "SPOILAGE DETECTED", (int(w_img*0.1), int(h_img*0.9)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                elif final_freshness == "Moderately Fresh":
                    cv2.rectangle(cv_img, (int(w_img*0.3), int(h_img*0.3)), (int(w_img*0.7), int(h_img*0.7)), (0, 220, 250), 3)
                    cv2.putText(cv_img, "EARLY RIPENING/SPOTS", (int(w_img*0.1), int(h_img*0.9)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 250), 2, lineType=cv2.LINE_AA)
                else:
                    # Fresh checkmark logo
                    cv2.putText(cv_img, "PASS: PREMIUM QUALITY", (int(w_img*0.1), int(h_img*0.9)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                    
                # Cache results for Tab 4 report generation
                st.session_state["last_scan"] = {
                    "fruit_type": fruit_selection,
                    "freshness": final_freshness,
                    "confidence": final_confidence,
                    "brix": pred_brix,
                    "moisture": pred_moisture,
                    "firmness": pred_firmness,
                    "spectrum": scan_data["spectrum"],
                    "gases": scan_data["gases"],
                    "days": sim_days,
                    "temp": sim_temp,
                    "img_bgr": cv_img
                }
                
                # Render Results
                res_col1, res_col2 = st.columns([2, 3])
                
                with res_col1:
                    st.image(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), caption="Vision Overlay Analysis", use_container_width=True)
                    
                    # Freshness Badge
                    if final_freshness == "Fresh":
                        badge_html = f"<div class='fresh-badge'>🍋 FRESH QUALITY ({final_confidence*100:.1f}%)</div>"
                    elif final_freshness == "Moderately Fresh":
                        badge_html = f"<div class='moderate-badge'>🍌 MODERATELY FRESH ({final_confidence*100:.1f}%)</div>"
                    else:
                        badge_html = f"<div class='spoiled-badge'>🍎 SPOILED / DISCARD ({final_confidence*100:.1f}%)</div>"
                        
                    st.markdown(f"**Freshness Status:** {badge_html}", unsafe_allow_html=True)
                    st.markdown("---")
                    
                    # Chemical attributes
                    st.markdown("##### 🧬 Internal Chemical Assessment (NIR)")
                    st.metric("Sugar Content (Brix)", f"{pred_brix:.2f} °Bx", delta=f"{pred_brix-11.5:.2f} relative to mean")
                    st.metric("Internal Moisture", f"{pred_moisture:.1f} %")
                    st.metric("Flesh Firmness", f"{pred_firmness:.1f} N")
                    
                with res_col2:
                    # 1. Plotly Spectrum Chart
                    spec_fig = go.Figure()
                    spec_fig.add_trace(go.Scatter(x=WAVELENGTHS, y=scan_data["spectrum"], name="Target Spectrum",
                                                 line=dict(color="#38BDF8", width=3)))
                    
                    # Highlight absorption bands
                    spec_fig.add_vrect(x0=660, x1=690, fillcolor="rgba(34, 197, 94, 0.1)", annotation_text="Chlorophyll", 
                                       annotation_position="top left", line_width=0)
                    spec_fig.add_vrect(x0=950, x1=990, fillcolor="rgba(56, 189, 248, 0.1)", annotation_text="Water absorption", 
                                       annotation_position="top left", line_width=0)
                    
                    spec_fig.update_layout(
                        title="Simulated Hyperspectral HSI Reflectance Curve",
                        xaxis_title="Wavelength (nm)",
                        yaxis_title="Reflectance (0 - 1)",
                        template="plotly_dark",
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(spec_fig, use_container_width=True)
                    
                    # 2. Gas Profile Bar Chart
                    gas_keys = list(scan_data["gases"].keys())
                    gas_vals = list(scan_data["gases"].values())
                    
                    gas_fig = go.Figure(data=[
                        go.Bar(name='Emitted Gas Levels (ppm)', x=['CO₂', 'Ethanol', 'Methane', 'Ammonia', 'Organic Volatiles'], 
                               y=gas_vals, marker_color='#818CF8')
                    ])
                    gas_fig.update_layout(
                        title="Electronic Nose (E-Nose) Volatile Gas Spectrum",
                        yaxis_title="Concentration (ppm)",
                        template="plotly_dark",
                        height=250,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(gas_fig, use_container_width=True)
            else:
                st.info("👈 Configure parameters and click 'Execute Non-Destructive Scan' to display results.")

# ==========================================
# TAB 3: ML EVALUATION LAB
# ==========================================
with tab3:
    st.markdown("### 📊 Machine Learning Model Analytics")
    st.markdown("Review validation scores and confusion metrics of the models trained on the simulated hyperspectral and sensor database.")
    
    if not os.path.exists("models/evaluation_metrics.json"):
        st.info("💡 Generate data and train models in the sidebar to review detailed performance metrics here.")
    else:
        with open("models/evaluation_metrics.json", "r") as f:
            metrics_data = json.load(f)
            
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='margin-bottom:0.25rem; font-size:0.9rem; color:#94A3B8;'>Spectral Model (HSI) Accuracy</p>
                    <h2 style='margin:0; color:#38BDF8;'>{metrics_data["spectral_classifier_accuracy"]*100:.2f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='margin-bottom:0.25rem; font-size:0.9rem; color:#94A3B8;'>E-Nose Grad Boost Accuracy</p>
                    <h2 style='margin:0; color:#818CF8;'>{metrics_data["enose_gb_accuracy"]*100:.2f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='margin-bottom:0.25rem; font-size:0.9rem; color:#94A3B8;'>Chemical Regressors (R² Mean)</p>
                    <h2 style='margin:0; color:#4ADE80;'>{np.mean([v["R2"] for v in metrics_data["spectral_regressor_metrics"].values()]):.4f}</h2>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        col_curve1, col_curve2 = st.columns(2)
        with col_curve1:
            st.markdown("##### Typical Spectral Profiles by Freshness Stage")
            # Build chart showing Fresh vs Spoiled typical reflectance curves
            curves_fig = go.Figure()
            
            # Wavelength sample
            w = WAVELENGTHS
            
            # Generate clean mathematical profiles for demonstration
            # Apple profiles
            for freshness, col in [("Fresh", "#4ADE80"), ("Moderately Fresh", "#FACC15"), ("Spoiled", "#F87171")]:
                # Mock average curves based on simulator logic
                if freshness == "Fresh":
                    r = np.clip(0.45 * np.exp(-((w - 650)/40)**2) + 0.75/(1+np.exp(-((w-730)/20))) - 0.25*np.exp(-((w-970)/40)**2) - 0.3*np.exp(-((w-675)/15)**2), 0.0, 1.0)
                elif freshness == "Moderately Fresh":
                    r = np.clip(0.45 * np.exp(-((w - 650)/40)**2) + 0.55/(1+np.exp(-((w-730)/20))) - 0.18*np.exp(-((w-970)/40)**2) - 0.12*np.exp(-((w-675)/15)**2), 0.0, 1.0)
                else:
                    r = np.clip(0.45 * np.exp(-((w - 650)/40)**2) + 0.35/(1+np.exp(-((w-730)/20))) - 0.08*np.exp(-((w-970)/40)**2) - 0.02*np.exp(-((w-675)/15)**2), 0.0, 1.0)
                
                curves_fig.add_trace(go.Scatter(x=w, y=r, name=freshness, line=dict(color=col, width=3.5)))
                
            curves_fig.update_layout(
                xaxis_title="Wavelength (nm)",
                yaxis_title="Reflectance",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(curves_fig, use_container_width=True)
            
        with col_curve2:
            st.markdown("##### Chemical Regressor Validation Performance (NIR)")
            metrics_table = []
            for target, vals in metrics_data["spectral_regressor_metrics"].items():
                metrics_table.append({
                    "Target Chemical Property": target.replace("_", " ").title(),
                    "Model R² Score (Variance Explained)": f"{vals['R2']:.4f}",
                    "Root Mean Squared Error (RMSE)": f"{vals['RMSE']:.4f}"
                })
            st.table(pd.DataFrame(metrics_table))

# ==========================================
# TAB 4: QUALITY REPORTS
# ==========================================
with tab4:
    st.markdown("### 📄 Freshness Quality Certificate")
    st.markdown("Compile and download an automated inspection report for your inspected batch.")
    
    last_scan = st.session_state.get("last_scan", None)
    
    if last_scan is None:
        st.info("💡 You must run a scan on Tab 2 before generating a certificate.")
    else:
        st.success(f"Inspection session found: **{last_scan['fruit_type']}** classified as **{last_scan['freshness']}**.")
        
        batch_id = st.text_input("Enter Batch ID", value="BATCH-2026-001")
        inspector_name = st.text_input("Inspector Name", value="Agent Antigravity")
        
        # Generator function for PDF
        def generate_pdf():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter,
                                    rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            story = []
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Heading1'],
                fontSize=22,
                textColor=colors.HexColor("#1E3A8A"),
                spaceAfter=15
            )
            h2_style = ParagraphStyle(
                'H2Style',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor("#2563EB"),
                spaceBefore=12,
                spaceAfter=6
            )
            body_style = ParagraphStyle(
                'BodyStyle',
                parent=styles['BodyText'],
                fontSize=10,
                leading=14,
                spaceAfter=6
            )
            
            # Document Title
            story.append(Paragraph("Non-Destructive Quality Certificate", title_style))
            story.append(Spacer(1, 10))
            
            # Metadata Grid Table
            meta_data = [
                [Paragraph("<b>Batch ID:</b>", body_style), Paragraph(batch_id, body_style),
                 Paragraph("<b>Scan Date:</b>", body_style), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), body_style)],
                [Paragraph("<b>Fruit Category:</b>", body_style), Paragraph(last_scan["fruit_type"], body_style),
                 Paragraph("<b>Inspector:</b>", body_style), Paragraph(inspector_name, body_style)],
                [Paragraph("<b>Storage Conditions:</b>", body_style), Paragraph(f"{last_scan['temp']}°C for {last_scan['days']} days", body_style),
                 Paragraph("<b>Analyzed Via:</b>", body_style), Paragraph("HSI, NIR & E-Nose Simulation", body_style)]
            ]
            t_meta = Table(meta_data, colWidths=[120, 150, 100, 150])
            t_meta.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t_meta)
            story.append(Spacer(1, 15))
            
            # Final Class
            status_color = "#16A34A" if last_scan["freshness"] == "Fresh" else "#CA8A04" if last_scan["freshness"] == "Moderately Fresh" else "#DC2626"
            story.append(Paragraph("Quality Assessment Verdict", h2_style))
            
            verdict_data = [
                [Paragraph("<b>Verdict / Freshness Class</b>", body_style), 
                 Paragraph(f"<font color='{status_color}'><b>{last_scan['freshness'].upper()}</b></font>", body_style)],
                [Paragraph("<b>ML Prediction Confidence</b>", body_style), 
                 Paragraph(f"{last_scan['confidence']*100:.2f}%", body_style)]
            ]
            t_verdict = Table(verdict_data, colWidths=[200, 320])
            t_verdict.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFFBEB")),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#FCD34D")),
                ('PADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(t_verdict)
            story.append(Spacer(1, 15))
            
            # Chemical Analysis Grid
            story.append(Paragraph("NIR Spectroscopy Chemical Prediction", h2_style))
            chem_data = [
                [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Reference Threshold</b>", body_style)],
                [Paragraph("Sugar Content", body_style), Paragraph(f"{last_scan['brix']:.2f} °Bx", body_style), Paragraph("10.0 - 15.0 °Bx", body_style)],
                [Paragraph("Internal Moisture", body_style), Paragraph(f"{last_scan['moisture']:.1f}%", body_style), Paragraph("70% - 90%", body_style)],
                [Paragraph("Flesh Firmness", body_style), Paragraph(f"{last_scan['firmness']:.1f} N", body_style), Paragraph("&gt; 20 N (Fresh)", body_style)]
            ]
            t_chem = Table(chem_data, colWidths=[180, 160, 180])
            t_chem.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#CBD5E1")),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t_chem)
            story.append(Spacer(1, 15))
            
            # Gas Sensor Grid
            story.append(Paragraph("E-Nose Volatile Compounds Analysis", h2_style))
            gas_data = [
                [Paragraph("<b>Gas Parameter</b>", body_style), Paragraph("<b>Sensor Level (ppm)</b>", body_style), Paragraph("<b>Risk Status</b>", body_style)],
                [Paragraph("CO<sub>2</sub> (MQ-135)", body_style), Paragraph(f"{last_scan['gases']['mq135']:.1f}", body_style), 
                 Paragraph("High" if last_scan['gases']['mq135'] > 800 else "Low", body_style)],
                [Paragraph("Ethanol (MQ-3)", body_style), Paragraph(f"{last_scan['gases']['mq3']:.1f}", body_style), 
                 Paragraph("Fermentation Warning" if last_scan['gases']['mq3'] > 150 else "Safe", body_style)],
                [Paragraph("Ripening gases (MQ-4)", body_style), Paragraph(f"{last_scan['gases']['mq4']:.1f}", body_style), 
                 Paragraph("Elevated" if last_scan['gases']['mq4'] > 80 else "Normal", body_style)],
            ]
            t_gas = Table(gas_data, colWidths=[180, 160, 180])
            t_gas.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#CBD5E1")),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t_gas)
            story.append(Spacer(1, 20))
            
            # Recommendations
            story.append(Paragraph("Storage & Shipping Instructions", h2_style))
            if last_scan["freshness"] == "Fresh":
                rec_text = "The inspected batch is of premium quality. Suitable for export and long-distance transport. Store in standard cold storage (4°C - 8°C, humidity 85%). Check in 5 days."
            elif last_scan["freshness"] == "Moderately Fresh":
                rec_text = "The batch shows initial signs of ripening/moisture reduction. Recommended for immediate local retail sale. Keep cold (4°C) and minimize transport shocks. Do not combine with new shipments."
            else:
                rec_text = "CRITICAL: The batch is spoiled. Remove immediately from storage to prevent ripening escalation and ethylene transmission to adjacent healthy stock. Clean containment area."
            story.append(Paragraph(rec_text, body_style))
            
            # Sign-off line
            story.append(Spacer(1, 30))
            story.append(Paragraph("___________________________", body_style))
            story.append(Paragraph(f"Authorized Digital Sign-off: <i>{inspector_name}</i>", body_style))
            
            doc.build(story)
            pdf_bytes = buffer.getvalue()
            buffer.close()
            return pdf_bytes
            
        pdf_data = generate_pdf()
        
        st.download_button(
            label="📥 Download Inspection PDF Certificate",
            data=pdf_data,
            file_name=f"Freshness_Quality_Report_{batch_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
