FROM python:3.10-slim

# System deps (opencv needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . /app

# Default command (RunPod can override, but you don't need to)
CMD ["python", "-u", "handler.py"]
