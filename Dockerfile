FROM python:3.10-slim

# Arbeitsverzeichnis setzen
WORKDIR /app

# System-Abhängigkeiten installieren (ffmpeg und libgl1 für OpenCV)
RUN apt-get update && apt-get install -y ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# LatentSync-Code kopieren
COPY . . 

# Kommando zum Starten des Handlers
CMD ["python", "-u", "handler.py"]
