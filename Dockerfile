FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files & enable stdout flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system packages required for pyarrow, pandas, and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Security: drop root privileges
RUN useradd -u 10001 -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8080
EXPOSE 8080

# Gunicorn
# -w: number of workers = 2*CPU+1 (overridable)
# -b: bind to all interfaces on port 8080
# Use app:server since Dash exposes the Flask server object
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:server"]
