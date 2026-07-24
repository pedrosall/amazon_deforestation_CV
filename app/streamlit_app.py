"""
AmazonNet - Frontend Streamlit.
Sube una imagen satelital, la envía al backend FastAPI y muestra
las predicciones, el overlay de Grad-CAM y las etiquetas de amenaza.
"""

import base64
import os

import requests
import streamlit as st

st.set_page_config(page_title="AmazonNet", page_icon="🛰️", layout="centered")

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

# Prioridad: st.secrets (Streamlit Cloud) > variable de entorno > localhost (dev local)
try:
    API_URL = st.secrets["API_URL"]
except (KeyError, FileNotFoundError):
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

THREAT_LABELS = {
    "artisinal_mine", "blow_down", "conventional_mine",
    "selective_logging", "slash_burn",
}

st.title("🛰️ AmazonNet")
st.caption("Clasificación multi-etiqueta de imágenes satelitales del Amazonas")

# --------------------------------------------------------------------------
# Subida de imagen
# --------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Sube una imagen satelital (JPEG, PNG o TIFF)",
    type=["jpg", "jpeg", "png", "tiff"],
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen subida", width="stretch")

    if st.button("Analizar imagen", type="primary"):
        with st.spinner("Analizando... (puede tardar unos segundos en CPU)"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as exc:
                st.error(f"Error al conectar con el backend: {exc}")
                st.stop()

        # ------------------------------------------------------------------
        # Resultado principal
        # ------------------------------------------------------------------
        st.subheader(f"Etiqueta principal: `{result['top_label']}`")

        threat_active = [label for label in result["threat_labels"]]
        if threat_active:
            st.error(f"⚠️ Amenazas detectadas: {', '.join(threat_active)}")

        # ------------------------------------------------------------------
        # Grad-CAM
        # ------------------------------------------------------------------
        st.subheader("Mapa de atención (Grad-CAM)")
        gradcam_bytes = base64.b64decode(result["gradcam_b64"])
        st.image(gradcam_bytes, caption=f"Regiones relevantes para '{result['top_label']}'", width="stretch")

        # ------------------------------------------------------------------
        # Probabilidades por etiqueta
        # ------------------------------------------------------------------
        st.subheader("Probabilidades por etiqueta")
        sorted_probs = sorted(result["probs"].items(), key=lambda x: x[1], reverse=True)

        for label, prob in sorted_probs:
            is_active = label in result["active_labels"]
            is_threat = label in THREAT_LABELS

            if is_threat and is_active:
                label_display = f":red[**{label}**] ⚠️"
            elif is_active:
                label_display = f"**{label}**"
            else:
                label_display = label

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(label_display)
                st.progress(min(prob, 1.0))
            with col2:
                st.markdown(f"{prob:.1%}")

else:
    st.info("Sube una imagen para empezar el análisis.")

st.divider()
st.caption(f"Backend: `{API_URL}`")