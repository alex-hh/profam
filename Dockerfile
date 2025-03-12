# Use official Python image as base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /workspace/profam

# Copy only requirements file first (to leverage Docker cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository
COPY . .

# Set a non-root user for security
RUN useradd -m devuser && chown -R devuser /workspace
USER devuser

# Set default command (optional, can be overridden)
CMD ["bash"]