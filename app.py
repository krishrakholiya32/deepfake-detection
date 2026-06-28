"""Deepfake Face Detector — Streamlit app.

Detects face-swap deepfakes in uploaded images using a self-trained
EfficientNet-B0 (ONNX) + MediaPipe face detection.

Scope: FaceForensics++ face-swap deepfakes only.
Out-of-scope: GAN-synthesised faces (DALL·E, Midjourney, etc.).
"""

import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

MODEL_PATH = Path("models/deepfake_effnetb0.onnx")
FACE_MARGIN = 0.30
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


@st.cache_resource(show_spinner=False)
def _load_detector():
    import mediapipe as mp
    return mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def extract_face(image_rgb: np.ndarray) -> np.ndarray | None:
    """Detect the most prominent face and return a 224×224 RGB crop, or None."""
    h, w = image_rgb.shape[:2]
    detector = _load_detector()
    results = detector.process(image_rgb)
    if not results.detections:
        return None

    det = results.detections[0]
    bb = det.location_data.relative_bounding_box
    mx = bb.width * FACE_MARGIN
    my = bb.height * FACE_MARGIN
    x1 = max(0, int((bb.xmin - mx) * w))
    y1 = max(0, int((bb.ymin - my) * h))
    x2 = min(w, int((bb.xmin + bb.width + mx) * w))
    y2 = min(h, int((bb.ymin + bb.height + my) * h))
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return np.array(Image.fromarray(crop).resize((FACE_SIZE, FACE_SIZE), Image.LANCZOS))


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
    "Upload a portrait photo to detect whether the face is **real** or "
    "**AI-manipulated** (face-swap)."
)

with st.expander("ℹ️ About this model"):
    st.markdown(
        "- Self-trained **EfficientNet-B0** on FaceForensics++ (C23 compression)\n"
        "- **Test AUC 0.9924 · Accuracy 96.2% · Precision 95.3% · Recall 97.2%**\n"
        "- Exported to ONNX; face detection via MediaPipe\n"
        "- **Scope:** Face-swap deepfakes only. GAN/diffusion-generated faces "
        "(Midjourney, DALL·E, Stable Diffusion) are a different artifact type "
        "and are out of scope — they may be classified as 'real'."
    )

uploaded = st.file_uploader("Upload a portrait image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(image)

    with st.spinner("Detecting face…"):
        face = extract_face(image_rgb)

    if face is None:
        st.warning("⚠️ No face detected. Please upload a clear portrait photo with a visible face.")
    else:
        with st.spinner("Running deepfake classifier…"):
            prob_fake = predict_fake_prob(face)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with col2:
            st.image(face, caption="Detected face (model input)", use_container_width=True)

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
