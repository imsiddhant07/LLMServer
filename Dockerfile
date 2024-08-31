ARG CUDA_IMAGE="12.5.0-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# Set working directory
WORKDIR /app

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY . .

# Setting build related env vars
# ENV CUDA_DOCKER_ARCH=all
# ENV GGML_CUDA=1

# # Install Python dependencies
# RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# # Install llama-cpp-python (build with cuda)
# RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Install additional Python dependencies from requirements.txt if needed
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
