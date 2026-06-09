import modal

app = modal.App("evoagent")

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.5.5")
    .dockerfile_commands("ENTRYPOINT []")
    .run_commands(
        "ln -sf /usr/bin/python3 /usr/bin/python",
        "pip3 install 'datasets>=2.18.0' 'google-genai>=1.0.0' 'pydantic>=2.5.0' "
        "'sentence-transformers>=2.7.0' 'matplotlib>=3.8.0' 'scipy>=1.12.0' "
        "'huggingface-hub>=0.22.0'",
    )
    .add_local_dir(".", remote_path="/evoagent")
)

volume = modal.Volume.from_name("evoagent-runs", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",
    timeout=21600,
    secrets=[modal.Secret.from_name("google"), modal.Secret.from_name("huggingface")],
    volumes={"/runs": volume},
)
def run():
    import os
    import subprocess
    os.chdir("/evoagent")
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

@app.function(
    image=image,
    gpu="T4",
    timeout=21600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/runs": volume},
)
def run_self():
    """Self-optimization: Qwen guides itself — no Gemini API needed."""
    import os
    import subprocess
    os.chdir("/evoagent")
    subprocess.run(
        [
            "python", "main.py",
            "--T", "5",
            "--dataset", "uitnlp/vimmrc2.0",
            "--output-dir", "/runs/exp_self",
            "--train-size", "100",
            "--model", "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "--self-optimize",
        ],
        check=True,
    )
    volume.commit()

@app.local_entrypoint()
def main():
    run.remote()

@app.local_entrypoint()
def self():
    run_self.remote()
