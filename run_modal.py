import modal

app = modal.App("evoagent")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "google-genai",
        "pydantic",
        "sentence-transformers",
        "matplotlib",
        "scipy",
        "huggingface-hub",
    )
    .add_local_dir(".", remote_path="/evoagent")
)

volume = modal.Volume.from_name("evoagent-runs", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    secrets=[modal.Secret.from_name("google"), modal.Secret.from_name("huggingface")],
    volumes={"/runs": volume},
)
def run():
    import os
    import subprocess
    os.chdir("/evoagent")
    # Clear previous run so we start fresh with real data
    import shutil
    if os.path.exists("/runs/exp01"):
        shutil.rmtree("/runs/exp01")
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    subprocess.run(
        [
            "python", "main.py",
            "--T", "5",
            "--dataset", "uitnlp/vimmrc2.0",
            "--output-dir", "/runs/exp01",
            "--train-size", "100",
            "--gemini-model", "gemini-2.5-flash",
            "--batch-size", "4",
        ],
        check=True,
        env=env,
    )
    volume.commit()

@app.local_entrypoint()
def main():
    run.remote()
