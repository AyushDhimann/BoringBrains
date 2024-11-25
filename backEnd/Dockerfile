# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements.txt .

# Resolve pip storage issue
RUN mkdir -p /pipcache /EASYOCR_MODEL_DIR
RUN TMPDIR=/pipcache pip install -r requirements.txt --cache-dir=/pipcache

# Copy the current directory contents into the container
COPY . .

# Command to run main.py
CMD ["python", "main.py"]

