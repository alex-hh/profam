# Use official Python image as base
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# Set the working directory inside the container
WORKDIR /workspace/profam

# Install Git
RUN apt-get update && apt-get install -y git

# Verify Git installation
RUN git --version

# Copy only requirements file first (to leverage Docker cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
# Disable if not interested in flash attention
# NOTE: update the URL to the release that matches torch, cuda, and python versions
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.6/flash_attn-2.7.4.post1+cu126torch2.6-cp311-cp311-linux_x86_64.whl
# Installation from scratch is very slow!
# RUN pip install flash-attn --no-build-isolation

# Copy the rest of the repository
COPY . .

# Set a non-root user for security
RUN useradd -m devuser && chown -R devuser /workspace
USER devuser

# Set default command (optional, can be overridden)
CMD ["bash"]