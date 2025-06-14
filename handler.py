import os, runpod, torch, tempfile, subprocess
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline

# 1) Modell nur EINMAL laden (außerhalb des Handlers)
pipe = LipsyncPipeline.from_pretrained(
    "ByteDance/LatentSync-1.6",
    torch_dtype=torch.float16
).to("cuda")

# 2) Job-Handler – wird von Runpod aufgerufen
def handler(job):
    """
    Erwartetes JSON-Eingabeformat:
    {
      "input": {
        "video_url": "https://…/person.mp4",
        "audio_url": "https://…/speech.wav",
        "steps": 25,          # optional
        "guidance": 2.0       # optional
      }
    }
    """
    inp = job["input"]
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, "video.mp4")
        aud = os.path.join(tmp, "audio.wav")
        # Dateien herunterladen
        subprocess.check_call(["curl", "-sL", inp["video_url"], "-o", vid])
        subprocess.check_call(["curl", "-sL", inp["audio_url"], "-o", aud])

        out_mp4 = os.path.join(tmp, "out.mp4")
        pipe(
            video_path=vid,
            audio_path=aud,
            video_out_path=out_mp4,
            num_inference_steps=inp.get("steps", 25),
            guidance_scale=inp.get("guidance", 2.0),
        )
        # Runpod lädt die Datei hoch und gibt eine https-URL zurück
        return runpod.serverless.upload_file(out_mp4)

runpod.serverless.start({"handler": handler})   # Pflichtaufruf
