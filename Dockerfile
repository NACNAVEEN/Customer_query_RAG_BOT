FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (saves ~1.5GB vs full PyTorch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models at build time (avoids download on every cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

# Copy application source code
COPY app/ ./app/
COPY web/ ./web/
COPY scripts/ ./scripts/
COPY .env.example ./.env

# Create data directories with correct permissions
RUN mkdir -p /app/data/uploads /app/data/vector_store /app/data/bm25_store /app/data/evaluation \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
