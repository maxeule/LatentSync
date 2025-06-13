FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy repository files
COPY . /app/

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt
RUN pip3 install runpod

# Download model checkpoints
RUN mkdir -p checkpoints/whisper && \
    wget -O checkpoints/latentsync_unet.pt https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/latentsync_unet_1.6.pt && \
    wget -O checkpoints/whisper/tiny.pt https://huggingface.co/ByteDance/LatentSync-1.5/resolve/main/whisper/tiny.pt

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the handler
CMD ["python3", "-u", "runpod_handler.py"]
