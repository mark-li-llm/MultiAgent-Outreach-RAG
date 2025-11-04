FROM continuumio/miniconda3:latest
WORKDIR /app

# Create conda environment
COPY envs/age.yaml .
RUN conda env create -f age.yaml && conda clean -afy

# Copy application code and data
COPY api/ api/
COPY scripts/ scripts/
COPY configs/ configs/
COPY data/vector/ data/vector/
COPY data/interim/ data/interim/

# Create runtime directories
RUN mkdir -p outputs state logs data/cache/embeddings

# Simple healthcheck (Railway expects this)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
    CMD wget -q --spider http://localhost:${PORT:-8000}/health || exit 1

# Run with conda environment (Railway sets $PORT)
CMD ["sh", "-c", "conda run -n age python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
