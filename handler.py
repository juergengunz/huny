import runpod
import torch
import base64
import io
import os
import requests
from PIL import Image
from transformers import AutoModelForCausalLM

# Set default device to CUDA to fix SigLIP2 attention_mask device mismatch
torch.set_default_device('cuda:0')

# --- GLOBAL MODEL LOADING (Executes once during Cold Start) ---
MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
# RunPod Model Cache path
CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
# Clean path without dots (required by Transformers)
CLEAN_MODEL_PATH = "/HunyuanImage-3-Instruct"

def get_model_path():
    """Locates the model in RunPod's cache and creates a symlink without dots."""
    if os.path.exists(CACHE_DIR):
        safe_name = f"models--{MODEL_NAME.replace('/', '--')}"
        full_path = os.path.join(CACHE_DIR, safe_name, "snapshots")
        if os.path.exists(full_path):
            snapshots = os.listdir(full_path)
            if snapshots:
                cache_path = os.path.join(full_path, snapshots[0])
                # Create symlink to path without dots (required by Transformers)
                if not os.path.exists(CLEAN_MODEL_PATH):
                    os.symlink(cache_path, CLEAN_MODEL_PATH)
                    print(f"--> Created symlink: {CLEAN_MODEL_PATH} -> {cache_path}")
                return CLEAN_MODEL_PATH
    return MODEL_NAME

model_path = get_model_path()
print(f"--> Loading model from: {model_path}")

kwargs = dict(
    attn_implementation="sdpa",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Explicit dtype for single GPU
    device_map="cuda:0",  # Force everything to single GPU
    moe_impl="eager",  # Use "flashinfer" if FlashInfer is installed
    moe_drop_tokens=True,
)

model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

# Workaround: Set model_version if missing (required by load_tokenizer due to path escaping issue)
if not hasattr(model.config, 'model_version'):
    model.config.model_version = "instruct"  # Default for Instruct/Distil models
    print(f"--> Set missing config.model_version = 'instruct'")

model.load_tokenizer(model_path)

# Ensure ALL model components are on CUDA (fixes SigLIP2 vision encoder device mismatch)
DEVICE = torch.device("cuda:0")
model = model.to(DEVICE)

# Explicitly move vision model if it exists (SigLIP2 encoder)
if hasattr(model, 'vision_model') and model.vision_model is not None:
    model.vision_model = model.vision_model.to(DEVICE)
    print(f"--> Moved vision_model to {DEVICE}")

# Also check for vision_encoder attribute
if hasattr(model, 'vision_encoder') and model.vision_encoder is not None:
    model.vision_encoder = model.vision_encoder.to(DEVICE)
    print(f"--> Moved vision_encoder to {DEVICE}")

# Ensure model is in eval mode
model.eval()
print(f"--> Model and Tokenizer ready on {next(model.parameters()).device}")

# --- UTILITIES ---

def is_url(s):
    """Check if a string is a URL."""
    return s.startswith(("http://", "https://"))

def load_image_from_url(url):
    """Download and load an image from a URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def base64_to_image(b64_str):
    """Convert a base64 string to a PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def load_image(image_input):
    """Load an image from either a URL or base64 string."""
    if is_url(image_input):
        return load_image_from_url(image_input)
    return base64_to_image(image_input)

def image_to_base64(img, fmt="PNG"):
    """Convert a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format=fmt.upper() if fmt.upper() != "JPG" else "JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- HANDLER ---

def handler(job):
    job_input = job["input"]
    prompt = job_input.get("prompt")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    seed = job_input.get("seed", 42)
    steps = job_input.get("steps", 8)
    out_format = job_input.get("format", "PNG")
    bot_task = job_input.get("bot_task")  # e.g. "think_recaption" for CoT reasoning, None by default
    drop_think = job_input.get("drop_think", False)  # Drop thinking output, default False
    
    # Check for image input (URL or Base64)
    image_input = job_input.get("image")
    input_image = load_image(image_input) if image_input else None
    
    # Use official generate_image method for Instruct/CoT
    # Wrap in inference_mode and set default device to avoid CPU/GPU tensor mismatch
    with torch.inference_mode(), torch.cuda.device(DEVICE):
        cot_text, samples = model.generate_image(
            prompt=prompt,
            image=input_image,
            seed=seed,
            image_size=f"{width}x{height}",
            use_system_prompt="en_unified",
            bot_task=bot_task,
            # drop_think=drop_think,
            diff_infer_steps=steps,
            verbose=2
        )

    # Return output in RunPod standard format
    return {
        "image": image_to_base64(samples[0], out_format),
        "reasoning": cot_text,
        "seed": seed
    }

runpod.serverless.start({"handler": handler})