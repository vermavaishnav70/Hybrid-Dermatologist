# ═══════════════════════════════════════════════════════════════════════════
#  Hybrid Skin Analysis System — Docker Container
# ═══════════════════════════════════════════════════════════════════════════
#
#  Build: docker build -f docker/Dockerfile -t hybrid-skin-analysis .
#  Run:   docker run --rm hybrid-skin-analysis
#  Demo:  docker run --rm -p 8501:8501 hybrid-skin-analysis streamlit run app/streamlit_app.py
#
#  This container provides full reproducibility: all dependencies are
#  pinned in requirements.txt and installed into a clean Python 3.10 image.

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create output directories
RUN mkdir -p outputs/phase1_baseline \
    outputs/phase2_deep_learning \
    outputs/phase3_hybrid \
    outputs/gpt4v_baseline

# Expose Streamlit port
EXPOSE 8501

# Default command: run evaluation
CMD ["make", "eval"]
