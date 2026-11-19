# Inference image for Render / local Docker
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KERAS_BACKEND=tensorflow \
    PREDICT_ROUTE=two \
    PORT=8080

WORKDIR /app

# System libs often needed by TensorFlow / Pillow wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY setup.py README.md requirements-prod.txt ./
COPY src ./src
COPY scripts ./scripts
COPY app.py ./
COPY templates ./templates

RUN pip install --upgrade pip \
    && pip install -r requirements-prod.txt

# Bake HF model into the image so runtime does not need Hub on every cold start
RUN python scripts/download_hf_model.py

EXPOSE 8080

# Render injects $PORT; default 8080 for local docker run
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 2 --timeout 120 app:app"]
