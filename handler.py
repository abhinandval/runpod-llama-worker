import runpod
from llama_cpp import Llama
import os
import subprocess

# Point to the RunPod Network Volume so the download is permanent
MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/gemma-4-26B-A4B-it-Q8_0.gguf")
# Clean URL without ?download=true
MODEL_URL = os.environ.get(
    "MODEL_URL", 
    "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q8_0.gguf"
)

def ensure_model_exists():
    """Checks for the model and downloads it safely if missing."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Starting download...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        # The -c flag ensures that if the pod crashes, the download resumes where it left off
        subprocess.run(["wget", "-c", "-O", MODEL_PATH, MODEL_URL], check=True)
        print("Download complete!")
    else:
        print(f"Model already exists at {MODEL_PATH}. Skipping download.")

# 1. Ensure model is downloaded before trying to load it
ensure_model_exists()

# 2. Load the model into VRAM/System RAM
print("Loading model into VRAM/RAM...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, # Offload all dense layers to GPU
    n_cpu_moe=30,    # Park 30 expert layers in system RAM
    n_ctx=8192,      # 8k context window
    type_k=8,        # Q8_0 Key Cache
    type_v=8,        # Q8_0 Value Cache
    flash_attn=True  # Flash Attention for maximum efficiency
)
print("Model loaded and ready.")

def handler(job):
    """Processes incoming API requests from RunPod."""
    job_input = job.get('input', {})
    prompt = job_input.get('prompt', '')
    max_tokens = job_input.get('max_tokens', 512)
    temperature = job_input.get('temperature', 0.7)
    
    if not prompt:
        return {"error": "Missing 'prompt' in the input JSON."}

    # Execute inference
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=False 
    )
    
    # Return the generated string
    return {"generated_text": response['choices'][0]['text']}

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})