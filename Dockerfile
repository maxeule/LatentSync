# 1. CUDA-fähiges Basis-Image (enthält libnvrtc.so.12 & CUDA Runtime)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

# 2. Non-interactive Mode für apt (verhindert tzdata-Prompts)
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# 3. Arbeitsverzeichnis
WORKDIR /app

# 4. System-Pakete installieren
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ffmpeg \
      libgl1 \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 5. Python-Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir runpod>=1.7.9 huggingface-hub onnxruntime-gpu

# 6. LatentSync-1.6 Checkpoints herunterladen
RUN huggingface-cli download ByteDance/LatentSync-1.6 \
      --local-dir /app/checkpoints \
      --exclude "*.git*" "README.md"

# 7. Symlinks für auxiliary Modelle
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    ln -s /app/checkpoints/auxiliary/2DFAN4-cd938726ad.zip \
         /root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip && \
    ln -s /app/checkpoints/auxiliary/s3fd-619a316812.pth \
         /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth && \
    ln -s /app/checkpoints/auxiliary/vgg16-397923af.pth \
         /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth

# 8. Restlichen Code kopieren
COPY . .

# 9. Config-Symlink für handler.py
RUN ln -s /app/configs/unet/stage2_efficient.yaml \
          /app/configs/unet/second_stage.yaml

# 10. Serverless Handler starten
CMD ["python", "-u", "handler.py"]
