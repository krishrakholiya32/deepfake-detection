# Deepfake Detection

A full-stack web application that detects whether a face in an image or video has been AI-manipulated (face-swap deepfake). Upload a photo or video clip and get a real/fake verdict powered by a self-trained EfficientNet-B0 classifier running as ONNX on CPU.

## Scope note

This model is trained on the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset (Deepfakes method — neural face swaps). It is designed to detect **face-swap deepfakes**, not GAN-synthesized faces (StyleGAN, Midjourney, DALL-E, etc.) or other AI-generated content. Those are a different artifact type and require different training data.

## Features

- Image upload (JPEG / PNG / WebP) — synchronous response
- Video upload (MP4 / MOV / AVI / WebM) — background job with polling
- Per-frame confidence chart for videos
- Full detection history
- Dockerized stack — one command to run

## Tech stack

| Layer | Technology |
|---|---|
| Model | EfficientNet-B0 (timm), exported to ONNX |
| Face detection | MTCNN (facenet-pytorch) |
| Backend | FastAPI, SQLAlchemy async, PostgreSQL, ONNX Runtime |
| Frontend | React 18, Vite, TypeScript, Tailwind CSS, Zustand, Recharts |
| Infra | Docker Compose, Nginx reverse proxy |
| Training | PyTorch, Kaggle Notebooks (GPU) |

## Architecture

```
browser
  └── nginx :8080
        ├── /api/*   → FastAPI backend :8000
        └── /*       → React SPA :80

backend
  ├── POST /api/detect/image        — classify image (sync)
  ├── POST /api/detect/video        — submit video (async, returns job_id)
  ├── GET  /api/detect/job/{job_id} — poll video job status
  ├── GET  /api/history             — list past detections
  └── GET  /health

models/
  └── deepfake_effnetb0.onnx   (gitignored — see Training)
```

## Quick start (Docker)

### Prerequisites

- Docker Desktop (or Docker + Docker Compose v2)
- Trained model file in `models/` (see [Training](#training))

```bash
git clone https://github.com/krishrakholiya32/deepfake-detection.git
cd deepfake-detection

# Copy .env.example and adjust if needed
cp .env.example .env

# Place deepfake_effnetb0.onnx in the project root, then:
python scripts/setup_models.py

docker-compose up --build
```

Open **http://localhost:8080** in your browser.

| Port | Service |
|---|---|
| 8080 | Main app (nginx) |
| 8001 | Backend API directly |
| 3001 | Frontend directly |

## Training

Training runs on Kaggle (free GPU). All scripts are in `training/scripts/`.

### 1. Set up a Kaggle Notebook

Create a new notebook and add the dataset:
- **Dataset**: [`xdxd003/ff-c23`](https://www.kaggle.com/datasets/xdxd003/ff-c23) (FaceForensics++, c23 compression, CC-BY-NC license)

Upload the contents of `training/` to the notebook (or clone the repo there).

### 2. Prepare the dataset

```bash
python training/scripts/prepare_dataset.py \
  --raw_dir /kaggle/input/ff-c23 \
  --output_dir /kaggle/working/data \
  --splits_dir training/configs/splits
```

Extracts face crops from video frames and splits by video ID (not randomly) to prevent identity leakage between train/val/test.

**Dataset layout**: `original/*.mp4` (real) and `Deepfakes/*.mp4` (fake) under a `FaceForensics++_C23/` folder. The official train/val/test splits (sourced from [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)) are pre-saved at `training/configs/splits/{train,val,test}.json`.

### 3. Train

```bash
python training/scripts/train.py --config training/configs/train_config.yaml
```

Two-phase fine-tuning:
1. Freeze all but the classification head
2. Unfreeze last 3 EfficientNet blocks, train with lower LR

Expected results: **AUC ~0.99, accuracy ~96%** on the video-ID-disjoint test split.

### 4. Export to ONNX

```bash
python training/scripts/export_model.py
```

Exports to ONNX (TorchScript exporter, `dynamo=False`) and verifies numerical agreement in probability space (tolerance 0.01).

### 5. Copy model to project

Download `deepfake_effnetb0.onnx` from the Kaggle output, place it in the project root, then:

```bash
python scripts/setup_models.py
```

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/detect/image` | Detect deepfake in an image. Form field: `file` (image) |
| `POST` | `/api/detect/video` | Submit a video for async detection. Form field: `file` (video) |
| `GET` | `/api/detect/job/{job_id}` | Poll video job status and results |
| `GET` | `/api/history` | List all past detections (newest first) |
| `GET` | `/api/history/{job_id}` | Single detection by job ID |
| `GET` | `/health` | Health check |

### Example — image

```bash
curl -X POST http://localhost:8001/api/detect/image \
  -F "file=@face.jpg"
```

```json
{
  "job_id": 1,
  "status": "done",
  "verdict": "real",
  "fake_probability": 0.0312
}
```

### Example — video

```bash
# Submit
curl -X POST http://localhost:8001/api/detect/video \
  -F "file=@clip.mp4"
# → {"job_id": 2, "status": "pending"}

# Poll until done
curl http://localhost:8001/api/detect/job/2
# → {"job_id": 2, "status": "done", "verdict": "fake", "fake_probability": 0.89, "frame_results": [...]}
```

**Verdict logic**: `fake` if `fake_probability >= DETECTION_THRESHOLD` (default 0.5), otherwise `real`. For videos, `fake_probability` is the mean across sampled frames.

If no face is detected in an image, the API returns HTTP 422 `"No face detected in image"`.

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `DETECTION_THRESHOLD` | `0.5` | Fake probability cutoff |
| `MAX_VIDEO_DURATION_SECS` | `300` | Maximum accepted video length (seconds) |
| `BACKEND_CPU_LIMIT` | `2` | Docker CPU quota for the backend container |
| `BACKEND_MEM_LIMIT` | `4G` | Docker memory limit for the backend container |

## Deployment

See [DEPLOY.md](DEPLOY.md) for step-by-step instructions to deploy on Oracle Cloud Always Free (ARM A1 VM).

## License

MIT — see [LICENSE](LICENSE).
