"""
app.py
------
Streamlit web application for the Plant Disease Detection System.

Launch
------
  streamlit run app.py

Features
--------
  • Upload a leaf image (JPG / PNG / BMP / WEBP).
  • Real-time CNN inference with confidence bar.
  • Disease details panel (name, plant, severity, treatment).
  • Grad-CAM heatmap overlay for visual explainability.
  • Download prediction result as a JSON report.
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
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
#  Cached Resource Loaders
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
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
#  Sidebar
# ─────────────────────────────────────────────

def render_sidebar():
    """Render sidebar controls and return user settings."""
    st.sidebar.title("⚙️ Settings")

    enable_gradcam = st.sidebar.toggle(
        "Show Grad-CAM Heatmap",
        value=False,
        help="Overlay a heatmap highlighting regions the model focused on.",
    )
    threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=1.0,
        value=float(CONFIDENCE_THRESHOLD),
        step=0.05,
        help="Predictions below this value are marked as 'Uncertain'.",
    )
    top_k = st.sidebar.number_input(
        "Top-K Predictions",
        min_value=1, max_value=10, value=3,
        help="Show the top K class predictions.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Model**: MobileNetV2 Transfer Learning (v2)\n\n"
        "**Classes**: 15 plant disease / healthy categories\n\n"
        "**Input size**: 224 × 224 px\n\n"
        "**Test Accuracy**: 89.63%"
    )

    return enable_gradcam, threshold, top_k


# ─────────────────────────────────────────────
#  Main App
# ─────────────────────────────────────────────

def main():
    # ── Header ───────────────────────────────────────────────────
    st.title(APP_TITLE)
    st.markdown(f"> {APP_DESCRIPTION}")
    st.markdown("---")

    enable_gradcam, threshold, top_k = render_sidebar()

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
            "Please ensure the model file is available or run training:\n"
            "```bash\n"
            "python train.py\n"
            "```"
        )
        model_loaded = False
    except Exception as exc:
        st.error(f"⚠️ Failed to load model: `{exc}`")
        model_loaded = False

    if not model_loaded or model is None:
        st.stop()

    # ── File Uploader ─────────────────────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("📤 Upload a Leaf Image")
        uploaded_file = st.file_uploader(
            label="Supported formats: JPG, JPEG, PNG, BMP, WEBP",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a clear, well-lit photo of a single plant leaf.",
        )

        if uploaded_file:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, caption="Uploaded Leaf Image", use_container_width=True)

    # ── Prediction ─────────────────────────────────────────────────
    with col_result:
        if uploaded_file:
            st.subheader("🔬 Prediction Results")

            with st.spinner("Analysing image …"):
                # Preprocess from bytes
                img_bytes  = uploaded_file.getvalue()
                img_array  = preprocess_image(img_bytes)              # (1, H, W, C)
                predictions = model.predict(img_array, verbose=0)[0]  # (num_classes,)

            top_indices     = np.argsort(predictions)[::-1][:top_k]
            top_class_idx   = int(top_indices[0])
            top_confidence  = float(predictions[top_class_idx])
            top_class_name  = class_names[top_class_idx]

            # ── Confidence indicator ──────────────────────────────
            is_uncertain = top_confidence < threshold
            status_icon  = "🟡" if is_uncertain else ("🟢" if "healthy" in top_class_name.lower() else "🔴")

            st.metric(
                label=f"{status_icon} Predicted Class",
                value=top_class_name.replace("___", " — ").replace("_", " "),
            )
            st.progress(
                min(top_confidence, 1.0),
                text=f"Confidence: {top_confidence * 100:.1f}%"
                     + (" ⚠️ (Uncertain)" if is_uncertain else ""),
            )

            # ── Top-K table ───────────────────────────────────────
            with st.expander(f"📊 Top {top_k} Predictions", expanded=False):
                rows = []
                for idx in top_indices:
                    rows.append({
                        "Class": class_names[idx].replace("___", " — ").replace("_", " "),
                        "Confidence": f"{predictions[idx] * 100:.2f}%",
                    })
                st.table(rows)

            # ── Disease Details ───────────────────────────────────
            details = get_disease_details(top_class_name, disease_df)

            with st.expander("🌱 Disease Information", expanded=True):
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.markdown(f"**Plant Species**: {details.get('plant_species', 'N/A')}")
                    st.markdown(f"**Disease Name** : {details.get('disease_name',  'N/A')}")
                    st.markdown(f"**Severity**     : {details.get('severity',      'N/A')}")
                with info_col2:
                    st.markdown("**Description**")
                    st.info(details.get("description", "No description available."))

                st.markdown("**Recommended Treatment / Management**")
                st.success(details.get("treatment", "Consult a certified agronomist."))

            # ── Grad-CAM ──────────────────────────────────────────
            if enable_gradcam:
                st.subheader("🔥 Grad-CAM Heatmap")
                try:
                    with st.spinner("Generating Grad-CAM heatmap …"):
                        heatmap = generate_gradcam(
                            model, img_array, pred_index=top_class_idx
                        )
                        blended = overlay_gradcam(pil_image, heatmap, alpha=0.45)
                        st.image(
                            blended,
                            caption="Grad-CAM: highlighted regions show areas most influential to the diagnosis",
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.warning(f"Grad-CAM generation failed: {exc}")

            # ── Download Report ───────────────────────────────────
            report = {
                "timestamp":    datetime.datetime.now().isoformat(),
                "image":        uploaded_file.name,
                "prediction":   top_class_name,
                "confidence":   round(top_confidence, 4),
                "uncertain":    is_uncertain,
                "disease_info": details,
                "top_k": [
                    {
                        "class": class_names[i],
                        "confidence": round(float(predictions[i]), 4),
                    }
                    for i in top_indices
                ],
            }
            st.download_button(
                label="⬇️ Download Prediction Report (JSON)",
                data=json.dumps(report, indent=2, ensure_ascii=False),
                file_name=f"prediction_{uploaded_file.name}.json",
                mime="application/json",
            )

        else:
            st.info("👆 Please upload a leaf image to get started.")

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:grey;'>"
        "Plant Disease Detection System · Built with TensorFlow/Keras + Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()

