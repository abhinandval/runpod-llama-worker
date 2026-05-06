# Use an official NVIDIA CUDA development image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set environment variables for compilation
ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

# Install required system tools and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the Python dependencies (compiles llama-cpp-python for CUDA)
RUN pip3 install --no-cache-dir runpod llama-cpp-python

# Copy the handler script
COPY handler.py /app/handler.py

# Start the worker
CMD ["python3", "handler.py"]