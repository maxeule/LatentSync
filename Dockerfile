# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Clone LatentSync repository
RUN git clone https://github.com/bytedance/LatentSync.git . && \
    git checkout main

# Install Python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
RUN pip3 install \
    diffusers==0.21.4 \
    transformers==4.35.2 \
    accelerate==0.24.1 \
    opencv-python==4.8.1.78 \
    pillow==10.1.0 \
    numpy==1.24.3 \
    scipy==1.11.4 \
    matplotlib==3.8.1 \
    tqdm==4.66.1 \
    omegaconf==2.3.0 \
    einops==0.7.0 \
    imageio[ffmpeg]==2.33.0 \
    av==11.0.0 \
    runpod==1.6.0

# Install additional dependencies for LatentSync
RUN pip3 install \
    face-alignment==1.3.5 \
    dlib==19.24.2 \
    mediapipe==0.10.8 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    pydub==0.25.1 \
    webrtcvad==2.0.10 \
    praat-parselmouth==0.4.3

# Install Whisper
RUN pip3 install openai-whisper==20230918

# Create necessary directories
RUN mkdir -p /app/checkpoints/whisper \
    && mkdir -p /app/outputs \
    && mkdir -p /app/temp

# Download model checkpoints
# Download LatentSync 1.6 UNet checkpoint
RUN wget -O /app/checkpoints/latentsync_unet.pt \
    "https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/latentsync_unet_1.6.pt" \
    || echo "Warning: Failed to download LatentSync 1.6 checkpoint"

# Download Whisper tiny model
RUN wget -O /app/checkpoints/whisper/tiny.pt \
    "https://huggingface.co/ByteDance/LatentSync-1.5/resolve/main/whisper/tiny.pt" \
    || echo "Warning: Failed to download Whisper model"

# Download Stable Diffusion VAE (required by LatentSync)
RUN python3 -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"

# Copy the handler file
COPY runpod_handler.py /app/runpod_handler.py

# Set permissions
RUN chmod -R 755 /app

# Verify installation
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Set the default command
CMD ["python3", "-u", "/app/runpod_handler.py"]
