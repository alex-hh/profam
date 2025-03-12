# Use official Python image as base
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

# Set the working directory inside the container
WORKDIR /workspace/profam

# Install Git
RUN apt-get update && apt-get install -y git

# Verify Git installation
RUN git --version

# Copy only requirements file first (to leverage Docker cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Disable if not interested in flash attention
RUN pip install flash-attn --no-build-isolation

# Copy the rest of the repository
COPY . .

# Set a non-root user for security
RUN useradd -m devuser && chown -R devuser /workspace
USER devuser

# Set default command (optional, can be overridden)
CMD ["bash"]