# Basis-Image mit Python 3.10
FROM python:3.10-slim

# Arbeitsverzeichnis im Container
WORKDIR /app

# 1. System-Abhängigkeiten installieren:
#    - build-essential (g++, make, etc.) für C/C++-Extensions (insightface)
#    - ffmpeg für Audio/Video-Verarbeitung
#    - libgl1 für OpenCV
RUN apt-get update && \
    apt-get install -y \
      build-essential \
      ffmpeg \
      libgl1 && \
    rm -rf /var/lib/apt/lists/*

# 2. Python-Abhängigkeiten aus requirements.txt installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Runpod-SDK hinzufügen (wird in handler.py importiert)
#    (optional, falls nicht bereits in requirements.txt)
RUN pip install --no-cache-dir runpod>=1.7.9

# 4. HuggingFace-Hub CLI installieren und LatentSync-1.6 Checkpoints herunterladen
RUN pip install --no-cache-dir huggingface-hub && \
    huggingface-cli download ByteDance/LatentSync-1.6 \
      --local-dir /app/checkpoints \
      --exclude "*.git*" "README.md"

# 5. Symlinks für Hilfs-Modelle setzen (Face-Detection, Sync-Net etc.)
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    ln -s /app/checkpoints/2DFAN4-cd938726ad.zip   /root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip && \
    ln -s /app/checkpoints/s3fd-619a316812.pth    /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth && \
    ln -s /app/checkpoints/vgg16-397923af.pth     /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth

# 6. Restlichen Code (inkl. handler.py und scripts/) in den Container kopieren
COPY . .

# 7. Serverless-Handler starten
CMD ["python", "-u", "handler.py"]
