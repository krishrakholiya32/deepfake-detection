# 🔍 Deepfake Face Detector

A Streamlit web app that detects whether a portrait photo contains a **face-swap deepfake** using a self-trained EfficientNet-B0 classifier exported to ONNX.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![AUC](https://img.shields.io/badge/Test_AUC-0.9924-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

**[🚀 Live Demo → deepfake-detection-zrik.streamlit.app](https://deepfake-detection-zrik.streamlit.app/)**

---

## ✨ Features

- 📸 Upload any portrait photo (JPG / PNG)
- 🚨 / ✅ Real or Fake verdict with probability score
- 📊 Visual confidence bar
- ⚡ Runs on CPU — no GPU needed

---

## 🧠 Scope Note

This model is trained on [FaceForensics++](https://github.com/ondyari/FaceForensics) (neural face-swap method, C23 compression). It detects **face-swap deepfakes** only. GAN/diffusion-generated faces (Midjourney, DALL·E, Stable Diffusion) are a different artifact type and are out of scope — they may be classified as real.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Web app framework |
| EfficientNet-B0 (ONNX) | Deepfake classification model |
| [ONNX Runtime](https://onnxruntime.ai) | CPU inference |
| [Pillow](https://python-pillow.org) | Image preprocessing |
| PyTorch + timm | Training (Kaggle GPU) |

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Test AUC | **0.9924** |
| Accuracy | **96.2%** |
| Precision | **95.3%** |
| Recall | **97.2%** |
| Dataset | FaceForensics++ C23 |
| Inference | ONNX Runtime (CPU) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/krishrakholiya32/deepfake-detection.git
cd deepfake-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ How It Works

1. Upload a portrait photo — the app center-crops it to a square and resizes to 224×224
2. ImageNet normalization is applied (mean/std)
3. The ONNX model runs inference on CPU (~50ms)
4. Logit is converted to fake probability via sigmoid
5. Verdict: **Fake** if probability ≥ 50%, else **Real**

---

## 📝 License

MIT — feel free to use, modify, and share.
