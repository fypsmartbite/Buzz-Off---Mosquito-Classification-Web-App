import os
import subprocess
import pathlib

MODEL_PATH = pathlib.Path("denseNet_161_64_25_0.001_adam.pt")
FILE_ID = os.environ.get("MODEL_FILE_ID")

if not FILE_ID:
    raise ValueError("MODEL_FILE_ID environment variable is not set. Please configure it in Render dashboard.")

if not MODEL_PATH.exists():
    print(f"Downloading model weights from Google Drive (ID: {FILE_ID})...")
    subprocess.run(
        ["gdown", f"https://drive.google.com/uc?id={FILE_ID}", "-O", str(MODEL_PATH)],
        check=True,
    )
    print(f"Model downloaded successfully to {MODEL_PATH}")
else:
    print(f"Model already exists at {MODEL_PATH}, skipping download.")
