import runpod
import base64
import subprocess
import os

# Handler-Funktion: wird für jede Anfrage aufgerufen
def handler(job):
    # 1. Eingabedaten aus dem Job extrahieren
    job_input = job.get("input", {})  # Input-Dictionary mit den Parametern
    video_b64 = job_input.get("video")    # Base64-codiertes Video (MP4)
    audio_b64 = job_input.get("audio")    # Base64-codiertes Audio (MP3)
    if not video_b64 or not audio_b64:
        return {"error": "Video oder Audio Input fehlt."}

    # 2. Base64-Daten in Dateien dekodieren und abspeichern
    try:
        video_bytes = base64.b64decode(video_b64)
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        return {"error": f"Base64-Decoding fehlgeschlagen: {e}"}

    input_video_path = "/tmp/input_video.mp4"
    input_audio_path = "/tmp/input_audio.mp3"
    with open(input_video_path, "wb") as vf:
        vf.write(video_bytes)
    with open(input_audio_path, "wb") as af:
        af.write(audio_bytes)

    # 3. Inferenz mit LatentSync durchführen (externes Skript aufrufen)
    output_video_path = "/tmp/output_video.mp4"
    command = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", "configs/unet/second_stage.yaml",
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--video_path", input_video_path,
        "--audio_path", input_audio_path,
        "--video_out_path", output_video_path,
        "--guidance_scale", "1.0"
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return {"error": f"Inferenz-Fehler (Subprocess): {e}"}
    except Exception as e:
        return {"error": f"Unbekannter Fehler bei Inferenz: {e}"}

    # 4. Ergebnis-Video einlesen und als Base64 kodieren
    if not os.path.exists(output_video_path):
        return {"error": "Ausgabedatei nicht gefunden. Prüfe Logs für Details."}
    with open(output_video_path, "rb") as of:
        output_bytes = of.read()
    output_b64 = base64.b64encode(output_bytes).decode('utf-8')

    # 5. Base64-Ergebnis zurückgeben
    return {"video": output_b64}

# Starten des Runpod-Serverless Handlers
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
