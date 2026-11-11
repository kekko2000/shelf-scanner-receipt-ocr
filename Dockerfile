FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_CACHE_HOME=/root/.paddlex
# OpenCV needs these runtime libraries on the slim Debian image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libxcb1 libgomp1 libstdc++6 libgcc-s1 libatomic1 curl wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and validate the Italian OCR models while building the image.
# Runtime startup then only initializes the already-cached models, avoiding
# a first-request download and the associated 502 risk.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='it', use_angle_cls=False)"

COPY app ./app

EXPOSE 8090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
