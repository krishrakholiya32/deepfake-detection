FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
# Install CPU-only torch first to keep the image small
RUN pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 # facenet-pytorch pins an old torch range (<2.3.0) that conflicts with the
 # pinned torch above; --no-deps is safe since it only needs torch/torchvision/numpy/Pillow,
 # all already installed.
 && pip install --no-cache-dir --no-deps facenet-pytorch==2.6.0

COPY backend/ .

# Models are mounted as a volume; uploads dir for temp video storage
RUN mkdir -p /app/uploads /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
