FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# System-Pakete
RUN apt-get update && \
    apt-get install -y ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Arbeits­verzeichnis
WORKDIR /app
COPY . /app

# Python-Abhängigkeiten
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install runpod

# Start­befehl für Runpod
CMD ["python", "-u", "handler.py"]
