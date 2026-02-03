import runpod
import torch
import base64
import io
import os
from PIL import Image
from transformers import AutoModelForCausalLM

# --- GLOBAL MODEL LOADING (Executes once during Cold Start) ---
MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct-Distil"
# RunPod Model Cache path
CACHE_DIR = "/runpod-volume/huggingface-cache/hub"

def get_model_path():
    """Locates the model in RunPod's cache or defaults to the model name."""
    if os.path.exists(CACHE_DIR):
        safe_name = f"models--{MODEL_NAME.replace('/', '--')}"
        full_path = os.path.join(CACHE_DIR, safe_name, "snapshots")
        if os.path.exists(full_path):
            snapshots = os.listdir(full_path)
            if snapshots:
                return os.path.join(full_path, snapshots[0])
    return MODEL_NAME

model_path = get_model_path()
print(f"--> Loading model from: {model_path}")

kwargs = dict(
    attn_implementation="sdpa",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    moe_impl="eager",
    moe_drop_tokens=True,
)

model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
model.load_tokenizer(model_path)
print("--> Model and Tokenizer ready.")

# --- UTILITIES ---

def base64_to_image(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def image_to_base64(img, fmt="PNG"):
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
    
    # Check for image input (Base64)
    b64_img = job_input.get("image")
    
    try:
        # Use official generate_image method for Instruct/CoT
        cot_text, samples = model.generate_image(
            prompt=prompt,
            image=base64_to_image(b64_img) if b64_img else None,
            seed=seed,
            image_size=f"{width}x{height}",
            use_system_prompt="en_unified",
            bot_task="think_recaption", # The "Think" reasoning step
            diff_infer_steps=steps,
            verbose=1
        )

        return {
            "image": image_to_base64(samples[0], out_format),
            "reasoning": cot_text,
            "seed": seed,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

runpod.serverless.start({"handler": handler})