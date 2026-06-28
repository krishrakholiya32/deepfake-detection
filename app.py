"""Deepfake Face Detector — Streamlit app.

Detects face-swap deepfakes in uploaded images using a self-trained
EfficientNet-B0 (ONNX). Center-crops the uploaded image to isolate the face region.

Scope: FaceForensics++ face-swap deepfakes only.
Out-of-scope: GAN-synthesised faces (DALL·E, Midjourney, etc.).
"""

import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

MODEL_PATH = Path("models/deepfake_effnetb0.onnx")
FACE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@st.cache_resource(show_spinner=False)
def _load_session():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"])


def preprocess(image: Image.Image) -> np.ndarray:
    """Center-square-crop then resize to 224×224."""
    w, h = image.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    cropped = image.crop((left, top, left + size, top + size))
    resized = cropped.resize((FACE_SIZE, FACE_SIZE), Image.LANCZOS)
    return np.array(resized)


def predict_fake_prob(face_rgb: np.ndarray) -> float:
    """Return P(fake) in [0, 1]."""
    session = _load_session()
    img = face_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    logits = session.run(["logits"], {"input": img})[0]
    prob_real = float(1.0 / (1.0 + np.exp(-logits[0, 0])))
    return 1.0 - prob_real


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Deepfake Detector", page_icon="🔍", layout="centered")

st.title("🔍 Deepfake Face Detector")
st.markdown(
    "Upload a **portrait photo** (face clearly visible) to detect whether it is "
    "**real** or **AI face-swapped**."
)

with st.expander("ℹ️ About this model"):
    st.markdown(
        "- Self-trained **EfficientNet-B0** on FaceForensics++ (C23 compression)\n"
        "- **Test AUC 0.9924 · Accuracy 96.2% · Precision 95.3% · Recall 97.2%**\n"
        "- Exported to ONNX · Preprocessing: center-square crop → 224×224\n"
        "- **Scope:** Face-swap deepfakes only. GAN/diffusion-generated faces "
        "(Midjourney, DALL·E, Stable Diffusion) are a different artifact type "
        "and are out of scope — they may be classified as 'real'.\n"
        "- **Tip:** Upload a tight portrait photo — the model works best when "
        "the face fills most of the frame."
    )

uploaded = st.file_uploader("Upload a portrait image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Preprocessing image…"):
        face_arr = preprocess(image)

    with st.spinner("Running deepfake classifier…"):
        prob_fake = predict_fake_prob(face_arr)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col2:
        st.image(face_arr, caption="Model input (224×224 crop)", use_container_width=True)

    st.divider()

    if prob_fake >= 0.5:
        st.error(f"### 🚨 FAKE — {prob_fake:.1%} fake probability")
        st.caption("This face shows characteristics consistent with face-swap manipulation.")
    else:
        st.success(f"### ✅ REAL — {1 - prob_fake:.1%} real probability")
        st.caption("No face-swap manipulation artifacts detected.")

    st.progress(prob_fake, text=f"Fake probability: {prob_fake:.1%}")

st.divider()
st.caption(
    "Model trained on FaceForensics++ (face-swap only) · "
    "Built with EfficientNet-B0 + ONNX Runtime · "
    "[GitHub](https://github.com/krishrakholiya32/deepfake-detection)"
)
