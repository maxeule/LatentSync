import os, tempfile, subprocess, runpod, torch, json
from huggingface_hub import snapshot_download
from pathlib import Path
from omegaconf import OmegaConf

# ------- 1. Einmaliger Modell-Download (≈ 7 GB) --------------------------
HF_REPO = "ByteDance/LatentSync-1.6"
CKPT_DIR = "/workspace/ls16"           # bleibt über Job-Starts erhalten

if not Path(CKPT_DIR).exists():
    # nur die großen checkpoints + YAML brauchen wir
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=CKPT_DIR,
        resume_download=True,
        allow_patterns=["*.pt", "*.pth", "*.yaml", "*.json", "*.ckpt"]
    )

# ------- 2. Komponenten der Pipeline laden ------------------------------
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from diffusers import AutoencoderKL, DDIMScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

# 2a) UNet
unet_cfg = OmegaConf.load(
    Path(__file__).parent / "latentsync" / "configs" / "unet" / "second_stage.yaml"
)
unet = UNet3DConditionModel.from_config(unet_cfg)
unet.load_state_dict(torch.load(f"{CKPT_DIR}/latentsync_unet.pt", map_location="cpu"), strict=False)

# 2b) SyncNet (zur Lipsync-Bewertung)
syncnet = torch.load(f"{CKPT_DIR}/stable_syncnet.pt", map_location="cpu")

# 2c) VAE & Scheduler – fertige Diffusers-Bausteine
vae        = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
scheduler  = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# 2d) Whisper-Encoder für Audio-Features
audio_feat = Audio2Feature(model_name="openai/whisper-small")

# ------- 3. Pipeline zusammensetzen & auf GPU schieben ------------------
pipe = LipsyncPipeline(
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    feature_extractor=audio_feat,
    syncnet=syncnet,
    torch_dtype=dtype,
).to(device).eval()

# ------- 4. Runpod-Handler ----------------------------------------------
def handler(job):
    """
    Erwartetes JSON:
    {
      "input": {
        "video_url": "...mp4",
        "audio_url": "...wav",
        "steps": 25,
        "guidance": 1.5
      }
    }
    """
    inp = job["input"]
    with tempfile.TemporaryDirectory() as tmp:
        vid = f"{tmp}/in.mp4"
        aud = f"{tmp}/in.wav"
        subprocess.check_call(["curl", "-sL", inp["video_url"], "-o", vid])
        subprocess.check_call(["curl", "-sL", inp["audio_url"], "-o", aud])

        out_mp4 = f"{tmp}/out.mp4"
        pipe(
            video_path       = vid,
            audio_path       = aud,
            video_out_path   = out_mp4,
            num_inference_steps = inp.get("steps", 25),
            guidance_scale      = inp.get("guidance", 1.5),
        )

        # Datei zu Runpod-Object-Store hochladen – gibt HTTPS-Link zurück
        return runpod.serverless.upload_file(out_mp4)

runpod.serverless.start({"handler": handler})
