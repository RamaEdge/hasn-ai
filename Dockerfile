# Multi-arch Python image for HASN-AI (API/Trainer/Monitor)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# System deps (minimal); uncomment if wheels are unavailable on your platform
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 8000

# Default to API; override command in K8s for trainer/monitor
CMD ["python", "src/api/main.py"]


