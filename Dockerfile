# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

ENV GGML_CUDA=on
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python \
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
