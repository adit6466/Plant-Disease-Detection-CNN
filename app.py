"""
app.py
------
Streamlit web application for the Plant Disease Detection System.
Enhanced with modern UI styling, glassmorphism, responsive cards,
agronomic advisory tabs, and Grad-CAM visual explainability.

Launch
------
  streamlit run app.py
"""

import json
import os
import sys
import io
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st
from PIL import Image

import importlib
import utils.image_processor
importlib.reload(utils.image_processor)

from config import (
    APP_TITLE, APP_DESCRIPTION,
    SAVED_MODEL_PATH, CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
)
from utils.model_builder import load_model
from utils.image_processor import (
    preprocess_image,
    generate_gradcam,
    overlay_gradcam,
)
from utils.disease_info import load_disease_info, get_disease_details
from utils.data_loader import get_class_names


# ─────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AgriVision AI — Plant Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
#  Custom CSS / Professional Theme Injection
# ─────────────────────────────────────────────

def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Global Typography & Background */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
                        radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.04) 0%, transparent 40%),
                        #0b0f19;
            color: #f3f4f6;
        }

        /* Hero Container */
        .hero-container {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 20px;
            padding: 32px 36px;
            margin-bottom: 28px;
            box-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.6), 0 0 20px rgba(16, 185, 129, 0.08);
            position: relative;
            overflow: hidden;
        }

        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 20% 20%, rgba(16, 185, 129, 0.08) 0%, transparent 50%);
            pointer-events: none;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.35);
            color: #34d399;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .hero-title {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1.2;
            background: linear-gradient(135deg, #ffffff 0%, #d1fae5 45%, #10b981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .hero-subtitle {
            color: #9ca3af;
            font-size: 1.05rem;
            line-height: 1.6;
            max-width: 850px;
            margin-bottom: 24px;
        }

        /* Metric Grid Pills */
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-top: 10px;
        }

        .stat-card {
            background: rgba(31, 41, 55, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 14px 18px;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }

        .stat-card:hover {
            border-color: rgba(16, 185, 129, 0.35);
            transform: translateY(-2px);
        }

        .stat-card-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #10b981;
            font-family: 'JetBrains Mono', monospace;
        }

        .stat-card-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }

        /* Modern Glass Card */
        .ui-card {
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 22px;
            box-shadow: 0 8px 30px -10px rgba(0, 0, 0, 0.5);
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }

        .ui-card:hover {
            border-color: rgba(16, 185, 129, 0.25);
            box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.6), 0 0 15px rgba(16, 185, 129, 0.06);
        }

        .ui-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 18px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }

        .ui-card-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f9fafb;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Status Banners */
        .status-banner {
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 18px;
            animation: fadeIn 0.4s ease-out;
        }

        .status-banner-healthy {
            background: linear-gradient(135deg, rgba(5, 150, 105, 0.22) 0%, rgba(16, 185, 129, 0.08) 100%);
            border: 1px solid rgba(16, 185, 129, 0.45);
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
        }

        .status-banner-diseased {
            background: linear-gradient(135deg, rgba(225, 29, 72, 0.22) 0%, rgba(244, 63, 94, 0.08) 100%);
            border: 1px solid rgba(244, 63, 94, 0.45);
            box-shadow: 0 4px 20px rgba(244, 63, 94, 0.15);
        }

        .status-banner-uncertain {
            background: linear-gradient(135deg, rgba(217, 119, 6, 0.22) 0%, rgba(245, 158, 11, 0.08) 100%);
            border: 1px solid rgba(245, 158, 11, 0.45);
            box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15);
        }

        .status-icon-circle {
            width: 52px;
            height: 52px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.8rem;
            flex-shrink: 0;
        }

        .status-icon-healthy {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        .status-icon-diseased {
            background: rgba(244, 63, 94, 0.2);
            color: #fb7185;
            border: 1px solid rgba(244, 63, 94, 0.35);
        }

        .status-icon-uncertain {
            background: rgba(245, 158, 11, 0.2);
            color: #fcd34d;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }

        .status-text-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #f9fafb;
            margin-bottom: 4px;
        }

        .status-text-sub {
            font-size: 0.9rem;
            color: #d1d5db;
        }

        /* Severity Pill Badges */
        .pill-high {
            background: rgba(239, 68, 68, 0.16);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.35);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .pill-medium {
            background: rgba(245, 158, 11, 0.16);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.35);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .pill-low {
            background: rgba(16, 185, 129, 0.16);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        /* Top-K Prediction Bar */
        .pred-bar-wrapper {
            margin-bottom: 12px;
        }

        .pred-bar-info {
            display: flex;
            justify-content: space-between;
            font-size: 0.88rem;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .pred-bar-bg {
            background: rgba(31, 41, 55, 0.7);
            border-radius: 9999px;
            height: 10px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .pred-bar-fill {
            height: 100%;
            border-radius: 9999px;
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Buttons Styling */
        .stButton>button {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 10px 24px !important;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.35) !important;
            transition: all 0.25s ease !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 22px rgba(16, 185, 129, 0.5) !important;
        }

        .stDownloadButton>button {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
            color: #38bdf8 !important;
            border: 1px solid rgba(56, 189, 248, 0.35) !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
            transition: all 0.25s ease !important;
            width: 100% !important;
        }

        .stDownloadButton>button:hover {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border-color: #0284c7 !important;
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45) !important;
            transform: translateY(-2px) !important;
        }

        /* Sidebar Brand Card */
        .sidebar-brand {
            padding: 16px;
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 16px;
            margin-bottom: 24px;
            text-align: center;
        }

        .sidebar-brand-title {
            font-size: 1.3rem;
            font-weight: 800;
            color: #f9fafb;
            margin-top: 6px;
        }

        .sidebar-brand-sub {
            font-size: 0.78rem;
            color: #34d399;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        /* Streamlit Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(31, 41, 55, 0.4);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            font-size: 0.9rem;
            color: #9ca3af;
            border: none;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(16, 185, 129, 0.2) !important;
            color: #34d399 !important;
            border: 1px solid rgba(16, 185, 129, 0.3) !important;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  Cached Resource Loaders
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing MobileNetV2 neural weights …")
def _load_model():
    """Load and cache the trained Keras model (runs once per session)."""
    return load_model(SAVED_MODEL_PATH)


@st.cache_data(show_spinner=False)
def _load_class_names():
    """Load and cache the list of class names."""
    return get_class_names()


@st.cache_data(show_spinner=False)
def _load_disease_df():
    """Load and cache the disease info DataFrame."""
    return load_disease_info()


# ─────────────────────────────────────────────
#  Sidebar Controls
# ─────────────────────────────────────────────

def render_sidebar():
    """Render modern sidebar controls and return user settings."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <span style="font-size: 2.2rem;">🌿</span>
            <div class="sidebar-brand-title">AgriVision AI</div>
            <div class="sidebar-brand-sub">Neural Agronomy Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### ⚙️ Diagnostic Settings")

    enable_gradcam = st.sidebar.toggle(
        "🔥 Show Grad-CAM Heatmap",
        value=True,
        help="Compute Gradient-weighted Class Activation Map to highlight diseased lesion regions on the leaf.",
    )

    threshold = st.sidebar.slider(
        "🎯 Confidence Threshold",
        min_value=0.20,
        max_value=0.95,
        value=float(CONFIDENCE_THRESHOLD),
        step=0.05,
        help="Predictions below this confidence rating will be flagged as Uncertain.",
    )

    top_k = st.sidebar.slider(
        "📊 Top-K Classes Displayed",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of top probabilistic predictions to compare.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧬 Architecture Specs")

    st.sidebar.markdown(
        """
        <div style="font-size: 0.84rem; color: #9ca3af; line-height: 1.8;">
            <div style="display:flex; justify-content:space-between;">
                <span>Backbone:</span> <b style="color:#f3f4f6;">MobileNetV2</b>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>Test Accuracy:</span> <b style="color:#10b981;">89.63%</b>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>Holdout Split:</span> <b style="color:#f3f4f6;">2,064 images</b>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>Resolution:</span> <b style="color:#f3f4f6;">224 × 224 px</b>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>Categories:</span> <b style="color:#38bdf8;">15 classes</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Quick Guide")
    st.sidebar.markdown(
        """
        1. 📸 **Upload** a clear photo of an affected leaf.
        2. ⚡ **Inference** identifies the disease and severity.
        3. 🔍 **Grad-CAM** visualizes exact lesion hotspots.
        4. 💊 **Action Plan** gives agronomic treatment steps.
        """
    )

    return enable_gradcam, threshold, top_k


# ─────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────

def main():
    inject_custom_css()
    enable_gradcam, threshold, top_k = render_sidebar()

    # ── Hero Section ─────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-badge">🌱 Deep Learning Diagnostic Engine v2.0</div>
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">
                High-precision convolutional neural network for automated crop pathology identification, 
                explainable lesion localization via Grad-CAM, and expert agronomic treatment advisory.
            </div>
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-card-value">89.63%</div>
                    <div class="stat-card-label">Test Set Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-card-value">15 Classes</div>
                    <div class="stat-card-label">Pepper, Potato, Tomato</div>
                </div>
                <div class="stat-card">
                    <div class="stat-card-value">MobileNetV2</div>
                    <div class="stat-card-label">Transfer Architecture</div>
                </div>
                <div class="stat-card">
                    <div class="stat-card-value">Grad-CAM</div>
                    <div class="stat-card-label">Visual Explainability</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model & Data Loading ─────────────────────────────────────
    model_loaded = True
    model = None
    class_names = []
    disease_df = None

    try:
        class_names = _load_class_names()
    except Exception as exc:
        st.error(f"⚠️ Failed to load class names: `{exc}`")
        model_loaded = False

    try:
        disease_df = _load_disease_df()
    except Exception as exc:
        st.error(f"⚠️ Failed to load disease database: `{exc}`")
        model_loaded = False

    try:
        model = _load_model()
    except FileNotFoundError as fnf:
        st.error(
            f"⚠️ **Trained model not found at `{SAVED_MODEL_PATH}`.**\n\n"
            f"Details: {fnf}\n\n"
            "Please ensure model weights are present in `models/`."
        )
        model_loaded = False
    except Exception as exc:
        st.error(f"⚠️ Failed to load neural model: `{exc}`")
        model_loaded = False

    if not model_loaded or model is None:
        st.stop()

    # ── Two-Column Operational Layout ─────────────────────────────
    col_upload, col_result = st.columns([1, 1.15], gap="large")

    # ═════════════════════════════════════════════════════════════
    #  LEFT COLUMN: Image Acquisition
    # ═════════════════════════════════════════════════════════════
    with col_upload:
        st.markdown(
            """
            <div class="ui-card">
                <div class="ui-card-header">
                    <div class="ui-card-title">
                        <span>📸</span> Leaf Image Acquisition
                    </div>
                    <span style="font-size:0.75rem; color:#9ca3af; background:rgba(255,255,255,0.06); padding:4px 10px; border-radius:6px;">
                        Input: 224 × 224 RGB
                    </span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            label="Upload Leaf Photo",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a clear, focused photograph of a single leaf under natural or bright light.",
            label_visibility="collapsed",
        )

        if uploaded_file:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(
                pil_image,
                caption=f"Uploaded: {uploaded_file.name} ({pil_image.width} × {pil_image.height}px)",
                use_container_width=True,
            )
        else:
            st.markdown(
                """
                <div style="text-align: center; padding: 36px 20px; border: 2px dashed rgba(16, 185, 129, 0.25); border-radius: 14px; background: rgba(31, 41, 55, 0.25);">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">🍃</div>
                    <div style="font-size: 1rem; font-weight: 700; color: #f3f4f6; margin-bottom: 6px;">
                        No leaf image uploaded yet
                    </div>
                    <div style="font-size: 0.85rem; color: #9ca3af; max-width: 380px; margin: 0 auto; line-height: 1.5;">
                        Drag and drop a photo or click <b>Browse files</b> above to begin diagnostic inference.
                    </div>
                    <div style="display: flex; justify-content: center; gap: 10px; margin-top: 18px; flex-wrap: wrap;">
                        <span style="font-size:0.75rem; background:rgba(16,185,129,0.1); color:#34d399; padding:4px 10px; border-radius:6px; border:1px solid rgba(16,185,129,0.25);">✓ Bell Pepper</span>
                        <span style="font-size:0.75rem; background:rgba(16,185,129,0.1); color:#34d399; padding:4px 10px; border-radius:6px; border:1px solid rgba(16,185,129,0.25);">✓ Potato</span>
                        <span style="font-size:0.75rem; background:rgba(16,185,129,0.1); color:#34d399; padding:4px 10px; border-radius:6px; border:1px solid rgba(16,185,129,0.25);">✓ Tomato</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════
    #  RIGHT COLUMN: Diagnostic Dashboard & Prescription
    # ═════════════════════════════════════════════════════════════
    with col_result:
        if uploaded_file:
            with st.spinner("Executing neural inference & calculating feature activations …"):
                img_bytes = uploaded_file.getvalue()
                img_array = preprocess_image(img_bytes)
                predictions = model.predict(img_array, verbose=0)[0]

            top_indices = np.argsort(predictions)[::-1][:top_k]
            top_class_idx = int(top_indices[0])
            top_confidence = float(predictions[top_class_idx])
            top_class_name = class_names[top_class_idx]

            details = get_disease_details(top_class_name, disease_df)
            is_healthy = details.get("is_healthy", False) or ("healthy" in top_class_name.lower())
            is_uncertain = top_confidence < threshold
            severity = str(details.get("severity", "medium")).lower()

            # Determine banner styling
            if is_uncertain:
                banner_class = "status-banner-uncertain"
                icon_class = "status-icon-uncertain"
                icon_sym = "⚠️"
                status_title = "Uncertain Diagnosis"
                status_sub = f"Confidence ({top_confidence * 100:.1f}%) is below your threshold ({threshold * 100:.0f}%). Verify image quality."
            elif is_healthy:
                banner_class = "status-banner-healthy"
                icon_class = "status-icon-healthy"
                icon_sym = "🌿"
                status_title = "Healthy Plant Detected"
                status_sub = f"The {details.get('plant_species', 'plant')} leaf shows no visible signs of pathogenic disease."
            else:
                banner_class = "status-banner-diseased"
                icon_class = "status-icon-diseased"
                icon_sym = "🔴"
                status_title = f"{details.get('disease_name', 'Pathology Detected')}"
                status_sub = f"Pathology identified on {details.get('plant_species', 'crop')} with high statistical certainty."

            # Severity Pill
            if severity == "high":
                sev_html = '<span class="pill-high">High Severity Risk</span>'
            elif severity == "low":
                sev_html = '<span class="pill-low">Low Severity Risk</span>'
            else:
                sev_html = '<span class="pill-medium">Moderate Severity Risk</span>'

            # ── Hero Status Banner ──────────────────────────────
            st.markdown(
                f"""
                <div class="status-banner {banner_class}">
                    <div class="status-icon-circle {icon_class}">
                        {icon_sym}
                    </div>
                    <div style="flex:1;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; flex-wrap:wrap; gap:6px;">
                            <span class="status-text-title">{status_title}</span>
                            {sev_html}
                        </div>
                        <div class="status-text-sub">{status_sub}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Confidence Gauge ────────────────────────────────
            st.markdown(
                f"""
                <div class="ui-card" style="padding: 18px 24px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 0.95rem; font-weight: 700; color: #f3f4f6;">Diagnostic Confidence Gauge</span>
                        <span style="font-size: 1.15rem; font-weight: 800; color: #10b981; font-family: 'JetBrains Mono', monospace;">
                            {top_confidence * 100:.2f}%
                        </span>
                    </div>
                    <div class="pred-bar-bg" style="height: 12px;">
                        <div class="pred-bar-fill" style="width: {min(top_confidence * 100, 100):.1f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Tabbed Agronomic Intelligence ───────────────────
            tab_pathology, tab_treatment, tab_distribution = st.tabs([
                "🔬 Disease Pathology",
                "💊 Treatment Protocol",
                "📊 Prediction Spectrum",
            ])

            with tab_pathology:
                st.markdown(
                    f"""
                    <div style="padding: 16px 4px;">
                        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:16px;">
                            <div style="background:rgba(31,41,55,0.4); padding:12px 16px; border-radius:10px; border:1px solid rgba(255,255,255,0.05);">
                                <div style="font-size:0.75rem; color:#9ca3af; text-transform:uppercase;">Host Species</div>
                                <div style="font-size:1.05rem; font-weight:700; color:#f3f4f6;">{details.get('plant_species', 'N/A')}</div>
                            </div>
                            <div style="background:rgba(31,41,55,0.4); padding:12px 16px; border-radius:10px; border:1px solid rgba(255,255,255,0.05);">
                                <div style="font-size:0.75rem; color:#9ca3af; text-transform:uppercase;">Identified Condition</div>
                                <div style="font-size:1.05rem; font-weight:700; color:#38bdf8;">{details.get('disease_name', 'N/A')}</div>
                            </div>
                        </div>
                        <div style="margin-bottom:12px;">
                            <span style="font-size:0.85rem; font-weight:700; color:#10b981;">📋 Symptom Description & Pathology:</span>
                            <p style="color:#d1d5db; font-size:0.92rem; line-height:1.6; margin-top:6px; background:rgba(31,41,55,0.3); padding:14px; border-radius:10px; border-left:3px solid #10b981;">
                                {details.get('description', 'No specific description recorded for this class.')}
                            </p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with tab_treatment:
                treatment_text = details.get('treatment', 'Consult a certified plant pathologist.')
                st.markdown(
                    f"""
                    <div style="padding: 16px 4px;">
                        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                            <div style="font-weight: 700; color: #34d399; font-size: 1rem; margin-bottom: 8px;">
                                🛡️ Recommended Agronomic Prescription:
                            </div>
                            <div style="color: #f3f4f6; font-size: 0.95rem; line-height: 1.65;">
                                {treatment_text}
                            </div>
                        </div>
                        <div style="font-size: 0.8rem; color: #9ca3af; line-height: 1.5;">
                            ⚠️ <i>Note: Always follow local pesticide safety regulations, ensure recommended dilution ratios, and rotate active chemical ingredients to prevent fungal resistance.</i>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with tab_distribution:
                st.markdown("<div style='padding: 16px 4px;'>", unsafe_allow_html=True)
                for idx in top_indices:
                    cls_name = class_names[idx].replace("___", " — ").replace("_", " ")
                    prob = float(predictions[idx]) * 100
                    st.markdown(
                        f"""
                        <div class="pred-bar-wrapper">
                            <div class="pred-bar-info">
                                <span style="color: #e5e7eb;">{cls_name}</span>
                                <span style="color: #34d399; font-family:'JetBrains Mono', monospace;">{prob:.2f}%</span>
                            </div>
                            <div class="pred-bar-bg">
                                <div class="pred-bar-fill" style="width: {prob:.1f}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Grad-CAM Visual Explainability Card ─────────────
            if enable_gradcam:
                st.markdown(
                    """
                    <div class="ui-card" style="margin-top: 20px;">
                        <div class="ui-card-header">
                            <div class="ui-card-title">
                                <span>🔥</span> Grad-CAM Visual Attention Map
                            </div>
                            <span style="font-size:0.75rem; color:#f59e0b; background:rgba(245,158,11,0.12); padding:4px 10px; border-radius:6px; border:1px solid rgba(245,158,11,0.25);">
                                Explainable AI
                            </span>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                try:
                    with st.spinner("Generating Grad-CAM explainability heatmap …"):
                        try:
                            heatmap = generate_gradcam(
                                model, img_array, None, top_class_idx
                            )
                        except TypeError:
                            heatmap = generate_gradcam(
                                model, img_array, pred_index=top_class_idx
                            )

                        blended = overlay_gradcam(pil_image, heatmap, alpha=0.45)

                        col_orig, col_cam = st.columns(2)
                        with col_orig:
                            st.image(pil_image, caption="Original Input Leaf", use_container_width=True)
                        with col_cam:
                            st.image(blended, caption="Grad-CAM Lesion Activation", use_container_width=True)

                        st.markdown(
                            """
                            <div style="font-size: 0.82rem; color: #9ca3af; line-height: 1.5; margin-top: 10px; padding: 10px 14px; background: rgba(31,41,55,0.4); border-radius: 8px;">
                                💡 <b>Interpretation</b>: The warm red and amber hotspots indicate the exact leaf tissue regions the MobileNetV2 neural backbone focused on to diagnose this condition.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                except Exception as exc:
                    st.warning(f"Grad-CAM generation failed: {exc}")

                st.markdown("</div>", unsafe_allow_html=True)

            # ── Download Agronomic Diagnostic Report ────────────
            report = {
                "system":       "AgriVision AI Diagnostic Report v2.0",
                "timestamp":    datetime.datetime.now().isoformat(),
                "image_name":   uploaded_file.name,
                "diagnosis": {
                    "class_name":       top_class_name,
                    "disease_name":     details.get("disease_name"),
                    "plant_species":    details.get("plant_species"),
                    "confidence_score": round(top_confidence, 4),
                    "is_healthy":       is_healthy,
                    "severity_level":   severity,
                    "uncertain_flag":   is_uncertain,
                },
                "advisory": {
                    "pathology_notes":  details.get("description"),
                    "treatment_steps":  details.get("treatment"),
                },
                "confidence_distribution": [
                    {
                        "rank": i + 1,
                        "class": class_names[idx],
                        "probability": round(float(predictions[idx]), 4),
                    }
                    for i, idx in enumerate(top_indices)
                ],
            }

            st.download_button(
                label="⬇️ Download Agronomic Diagnostic Report (JSON)",
                data=json.dumps(report, indent=2, ensure_ascii=False),
                file_name=f"agrivision_report_{uploaded_file.name.split('.')[0]}.json",
                mime="application/json",
            )

        else:
            # Standby state placeholder
            st.markdown(
                """
                <div class="ui-card" style="text-align: center; padding: 60px 30px;">
                    <div style="font-size: 3.5rem; margin-bottom: 16px;">🔬</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #f9fafb; margin-bottom: 8px;">
                        Awaiting Leaf Input
                    </div>
                    <div style="font-size: 0.95rem; color: #9ca3af; max-width: 440px; margin: 0 auto; line-height: 1.6;">
                        Upload a photo of a plant leaf on the left panel. The neural engine will instantly run inference, 
                        display the disease diagnosis, generate Grad-CAM explainability maps, and outline agronomic treatments.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Modern Footer ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; color: #6b7280; font-size: 0.85rem; padding: 8px 4px; flex-wrap: wrap; gap: 10px;">
            <div>
                🌿 <b>AgriVision AI</b> · Built with TensorFlow/Keras & Streamlit
            </div>
            <div>
                Clean Stratified Holdout Test Accuracy: <b style="color: #10b981;">89.63%</b>
            </div>
            <div>
                <a href="https://github.com/adit6466/Plant-Disease-Detection-CNN" target="_blank" style="color: #38bdf8; text-decoration: none;">
                    GitHub Repository ↗
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()
