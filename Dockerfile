# Use RunPod's optimized PyTorch base
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y git git-lfs && git lfs install

# Install Python requirements
# Note: FlashInfer and FlashAttention are highly recommended for MoE speed
RUN pip install --upgrade pip
RUN pip install runpod transformers accelerate diffusers Pillow requests

# Copy your handler
COPY handler.py /handler.py

# Optional: Pre-download model into the image if NOT using RunPod Cache
# RUN python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('tencent/HunyuanImage-3.0-Instruct-Distil', trust_remote_code=True)"

CMD [ "python3", "-u", "/handler.py" ]