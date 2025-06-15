FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ---------- Linux-Pakete -------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

# ---------- Arbeitsverzeichnis & Code -----------------------------------
WORKDIR /app
COPY . /app

# ---------- Python-Pakete ------------------------------------------------
# 1) LatentSync direkt von GitHub (enthält Code + YAML-Konfigs)
# 2) Runpod-SDK fürs Serverless-Runtime
RUN pip install --upgrade pip && \
    pip install \
        git+https://github.com/bytedance/LatentSync.git \
        runpod

# ---------- Start-Befehl -------------------------------------------------
CMD ["python", "-u", "handler.py"]
