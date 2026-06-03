import modal

app = modal.App("evoagent")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch",
        "vllm==0.5.5",
        "transformers",
        "accelerate",
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
    import shutil
    if os.path.exists("/runs/exp01"):
        shutil.rmtree("/runs/exp01")
    subprocess.run(
        [
            "python", "main.py",
            "--T", "5",
            "--dataset", "uitnlp/vimmrc2.0",
            "--output-dir", "/runs/exp01",
            "--train-size", "100",
            "--model", "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "--gemini-model", "gemini-2.5-flash",
        ],
        check=True,
    )
    volume.commit()

@app.local_entrypoint()
def main():
    run.remote()
