# Use RunPod's PyTorch base with CUDA 12.8 for B200/Blackwell support
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y git git-lfs && git lfs install

# Install Python requirements
# Note: FlashInfer and FlashAttention are highly recommended for MoE speed
RUN pip install --upgrade pip

# Copy requirements file and install dependencies
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Copy your handler
COPY handler.py /handler.py

CMD [ "python3", "-u", "/handler.py" ]