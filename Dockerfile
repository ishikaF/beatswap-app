FROM python:3.11-slim
FROM mirror.gcr.io/library/python:3.11-slim


# System packages for audio + building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["bash", "-lc", "streamlit run app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]
