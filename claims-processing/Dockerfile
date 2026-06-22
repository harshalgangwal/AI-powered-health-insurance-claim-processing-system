FROM python:3.11-slim

# System dependencies
# - poppler-utils: required by pdf2image to convert PDF pages to images
# - libgomp1:      required by PaddlePaddle (OpenMP runtime)
# - libglib2.0-0:  required by OpenCV (used internally by PaddleOCR)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download PaddleOCR models at build time so the container starts fast.
# The model download is triggered by a dummy run during the build.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', show_log=False)"

# Copy application source
COPY . .

# Create directories that the app writes to at runtime
RUN mkdir -p /app/uploads /app/data

# SQLite database lives in /app/data (mapped to a volume in compose)
ENV DATABASE_URL="sqlite:////app/data/claims.db"
ENV UPLOAD_DIR="/app/uploads"
ENV POLICY_PATH="/app/policy_terms.json"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
