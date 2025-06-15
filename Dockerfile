# Basis-Image mit Python 3.10
FROM python:3.10-slim

# Arbeitsverzeichnis im Container
WORKDIR /app

# 1. System-Abhängigkeiten installieren (ffmpeg für Audio/Video, libgl1 für OpenCV)
RUN apt-get update && \
    apt-get install -y ffmpeg libgl1 && \
    rm -rf /var/lib/apt/lists/*

# 2. Python-Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. HuggingFace-Hub CLI installieren und LatentSync-1.6 Checkpoints herunterladen
RUN pip install huggingface-hub && \
    huggingface-cli download ByteDance/LatentSync-1.6 \
      --local-dir /app/checkpoints \
      --exclude "*.git*" "README.md"

# 4. Symlinks für Hilfs-Modelle (Face-Detection, Sync-Net etc.)
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    ln -s /app/checkpoints/2DFAN4-cd938726ad.zip /root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip && \
    ln -s /app/checkpoints/s3fd-619a316812.pth      /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth && \
    ln -s /app/checkpoints/vgg16-397923af.pth      /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth

# 5. Restlichen Code (inkl. handler.py) in den Container kopieren
COPY . .

# 6. Serverless-Handler starten
CMD ["python", "-u", "handler.py"]
