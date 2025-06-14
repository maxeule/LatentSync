import runpod
import torch
import os
import tempfile
import requests
import logging
from pathlib import Path
import subprocess
import json
import base64
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LatentSync components - Updated for newer versions
import sys
sys.path.append('/app')

try:
    # Try newer import structure first
    from diffusers import AutoencoderKL, DDIMScheduler
    from diffusers.schedulers import PNDMScheduler
    
    # Import video processing
    import decord
    decord.bridge.set_bridge('torch')
    
    # Additional optimizations if available
    try:
        from DeepCache import DeepCacheSD
        DEEPCACHE_AVAILABLE = True
    except ImportError:
        DEEPCACHE_AVAILABLE = False
        logger.info("DeepCache not available, using standard inference")
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Fallback imports for compatibility
    pass

from src.models.unet import UNet3DConditionModel
from src.models.whisper import whisper
from src.pipelines.latentsync_pipeline import LatentSyncPipeline
from src.utils.config import load_config
from src.utils.utils import tensor2video
import cv2
import numpy as np

# Global variables for model caching
pipeline = None
device = None

def download_file(url, destination):
    """Download a file from URL to destination"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded file from {url} to {destination}")
        return True
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return False

def upload_to_temp_host(file_path):
    """Upload file to a temporary host and return URL"""
    try:
        # Using file.io as a temporary file host (expires after 24 hours)
        with open(file_path, 'rb') as f:
            response = requests.post(
                'https://file.io',
                files={'file': f}
            )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('link')
        else:
            logger.error(f"Failed to upload file: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return None

def encode_video_to_base64(video_path):
    """Encode video file to base64 string"""
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        return base64.b64encode(video_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding video: {str(e)}")
        return None

def initialize_models():
    """Initialize LatentSync models"""
    global pipeline, device
    
    if pipeline is not None:
        return pipeline
    
    logger.info("Initializing LatentSync models...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config_path = "/app/configs/unet/stage2.yaml"
    if not os.path.exists(config_path):
        config_path = "/app/configs/unet/stage1.yaml"
    
    config = load_config(config_path)
    
    # Initialize models
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    # Load Whisper model
    whisper_model = whisper.load_model(
        "/app/checkpoints/whisper/tiny.pt",
        device=device
    )
    
    # Load UNet
    unet = UNet3DConditionModel.from_config(config.model.unet)
    checkpoint_path = "/app/checkpoints/latentsync_unet.pt"
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        unet.load_state_dict(checkpoint, strict=False)
        logger.info("Loaded UNet checkpoint")
    else:
        logger.warning("UNet checkpoint not found!")
    
    unet = unet.to(device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    
    # Initialize scheduler
    scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1
    )
    
    # Create pipeline
    pipeline = LatentSyncPipeline(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        whisper_model=whisper_model
    )
    
    logger.info("Models initialized successfully")
    return pipeline

def process_video_with_latentsync(video_path, audio_path, inference_steps=30, guidance_scale=2.0):
    """Process video with LatentSync"""
    try:
        pipeline = initialize_models()
        
        # Try to use decord for better performance
        try:
            from decord import VideoReader
            vr = VideoReader(video_path)
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            width, height = vr[0].shape[1], vr[0].shape[0]
            logger.info(f"Using decord - Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
            use_decord = True
        except:
            # Fallback to OpenCV
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Using OpenCV - Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
            use_decord = False
        
        # Prepare output video
        output_path = video_path.replace('.mp4', '_synced.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))
        
        # Process video in chunks (25 frames at a time as per LatentSync)
        chunk_size = 25
        
        for start_frame in range(0, total_frames, chunk_size):
            end_frame = min(start_frame + chunk_size, total_frames)
            
            # Read frames
            frames = []
            
            if use_decord:
                # Use decord for frame reading
                frame_indices = list(range(start_frame, end_frame))
                batch_frames = vr.get_batch(frame_indices).asnumpy()
                for frame in batch_frames:
                    # Resize to 512x512 (LatentSync 1.6 resolution)
                    frame = cv2.resize(frame, (512, 512))
                    frames.append(frame)
            else:
                # Use OpenCV
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for _ in range(end_frame - start_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize to 512x512 (LatentSync 1.6 resolution)
                    frame = cv2.resize(frame, (512, 512))
                    frames.append(frame)
            
            if len(frames) == 0:
                break
            
            # Convert frames to tensor
            frames_tensor = torch.stack([
                torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                for frame in frames
            ]).to(device)
            
            # Run LatentSync pipeline
            with torch.no_grad():
                output_frames = pipeline(
                    frames=frames_tensor,
                    audio_path=audio_path if audio_path else video_path,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    start_time=start_frame / fps,
                    end_time=end_frame / fps
                )
            
            # Write output frames
            for frame in output_frames:
                frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                out.write(frame_np)
            
            logger.info(f"Processed frames {start_frame} to {end_frame}")
        
        # Clean up
        if not use_decord:
            cap.release()
        out.release()
        
        # Add audio to output video
        if audio_path and audio_path != video_path:
            output_with_audio = output_path.replace('.mp4', '_final.mp4')
            cmd = [
                'ffmpeg', '-i', output_path, '-i', audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-shortest',
                output_with_audio, '-y'
            ]
        else:
            output_with_audio = output_path.replace('.mp4', '_final.mp4')
            cmd = [
                'ffmpeg', '-i', output_path, '-i', video_path,
                '-c:v', 'copy', '-map', '0:v', '-map', '1:a',
                '-shortest', output_with_audio, '-y'
            ]
        
        subprocess.run(cmd, check=True)
        
        return output_with_audio
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

def handler(job):
    """
    RunPod handler function for LatentSync 1.6
    
    Expected input format:
    {
        "input": {
            "video_url": "https://example.com/video.mp4",
            "audio_url": "https://example.com/audio.wav" (optional),
            "inference_steps": 30,
            "guidance_scale": 2.0,
            "return_type": "url" or "base64"
        }
    }
    """
    try:
        logger.info(f"Received job: {job['id']}")
        
        # Parse input
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        audio_url = job_input.get("audio_url")
        inference_steps = job_input.get("inference_steps", 30)
        guidance_scale = job_input.get("guidance_scale", 2.0)
        return_type = job_input.get("return_type", "url")
        
        # Validate input
        if not video_url:
            return {"error": "video_url is required"}
        
        # Validate parameters
        if not isinstance(inference_steps, (int, float)) or inference_steps < 1:
            inference_steps = 30
        
        if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0:
            guidance_scale = 2.0
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download video
            video_path = os.path.join(temp_dir, "input_video.mp4")
            if not download_file(video_url, video_path):
                return {"error": "Failed to download video"}
            
            # Download audio if provided
            audio_path = None
            if audio_url and audio_url != video_url:
                audio_path = os.path.join(temp_dir, "input_audio.wav")
                if not download_file(audio_url, audio_path):
                    logger.warning("Failed to download audio, using video audio")
                    audio_path = None
            
            # Process video
            logger.info("Starting LatentSync processing...")
            output_path = process_video_with_latentsync(
                video_path,
                audio_path,
                inference_steps=int(inference_steps),
                guidance_scale=float(guidance_scale)
            )
            
            # Prepare output
            result = {
                "status": "completed",
                "message": "Video processed successfully"
            }
            
            if return_type == "base64":
                # Return base64 encoded video
                video_base64 = encode_video_to_base64(output_path)
                if video_base64:
                    result["output"] = video_base64
                    result["output_type"] = "base64"
                else:
                    return {"error": "Failed to encode output video"}
            else:
                # Upload to temporary host and return URL
                output_url = upload_to_temp_host(output_path)
                if output_url:
                    result["output_url"] = output_url
                    result["output_type"] = "url"
                else:
                    # Fallback to base64 if upload fails
                    video_base64 = encode_video_to_base64(output_path)
                    if video_base64:
                        result["output"] = video_base64
                        result["output_type"] = "base64"
                    else:
                        return {"error": "Failed to prepare output"}
            
            # Add metadata
            result["metadata"] = {
                "inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "model_version": "LatentSync 1.6",
                "resolution": "512x512"
            }
            
            logger.info(f"Job completed successfully: {job['id']}")
            return result
            
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "status": "failed"
        }

# Start the RunPod serverless worker
if __name__ == "__main__":
    logger.info("Starting LatentSync 1.6 RunPod worker...")
    runpod.serverless.start({"handler": handler})
