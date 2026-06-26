# Deployment — Oracle Cloud Always Free (ARM A1)

Step-by-step guide to deploy on an Oracle Cloud Always Free ARM VM (4 OCPU / 24 GB RAM, no expiry).

## 1. Provision the VM

1. Sign up at [cloud.oracle.com](https://cloud.oracle.com) (Always Free tier, no credit card charge).
2. Create a **Compute Instance**:
   - Shape: `VM.Standard.A1.Flex` — set **4 OCPU / 24 GB RAM** (free quota).
   - Image: **Ubuntu 22.04 Minimal** (aarch64).
   - Add your SSH public key.
3. In **Networking → Security Lists**, open inbound TCP on port **8080** (or 80 if you alias it).

## 2. Install Docker on the VM

```bash
ssh ubuntu@<VM_PUBLIC_IP>

sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg
sudo install -m0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Run Docker without sudo
sudo usermod -aG docker ubuntu
newgrp docker
```

## 3. Clone the repo and add the model

```bash
git clone https://github.com/krishrakholiya32/deepfake-detection.git
cd deepfake-detection
```

Copy `deepfake_effnetb0.onnx` from your Kaggle training output to the VM:

```bash
# From your local machine:
scp deepfake_effnetb0.onnx ubuntu@<VM_PUBLIC_IP>:~/deepfake-detection/
```

Then on the VM:

```bash
python3 scripts/setup_models.py
```

## 4. Configure environment

```bash
cp .env.example .env
# Edit SERVER_IP to your VM's public IP
nano .env
```

Set `SERVER_IP=<VM_PUBLIC_IP>` so the frontend can reach the backend correctly.

## 5. Build and start

```bash
docker compose up --build -d
```

This builds the backend and frontend images, starts PostgreSQL, and launches nginx on port **8080**.

Check all containers are healthy:

```bash
docker compose ps
```

## 6. Verify

```bash
curl http://localhost:8001/health
# → {"status":"ok","version":"1.0.0"}

curl http://localhost:8080
# → HTML of the React app
```

Open `http://<VM_PUBLIC_IP>:8080` in your browser.

## 7. Persistence

PostgreSQL data and uploaded videos are stored in named Docker volumes (`pgdata`, `uploads`) and survive container restarts. To restart after a VM reboot:

```bash
cd ~/deepfake-detection && docker compose up -d
```

To auto-start on boot:

```bash
sudo systemctl enable docker
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker: command not found` | Docker not installed | Re-run step 2 |
| Backend exits immediately | Model file missing | Run `setup_models.py`, check `models/` |
| `connection refused` on port 8080 | OCI security list | Open TCP 8080 in the OCI console |
| Frontend can't reach `/api/` | `SERVER_IP` not set | Set `SERVER_IP` in `.env`, re-run `docker compose up -d` |
| `No space left on device` | VM disk full | Run `docker system prune -f` |
| Video job stuck in `pending` | Background task crashed | Check `docker compose logs backend` for traceback |
