FROM python:3.11-slim-bookworm

# Install system dependencies and Node.js
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    git \
    curl \
    ffmpeg \
    flac \
    libsndfile1 \
    sox \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone kiwiclient from GitHub (used directly, not installed as package)
# kiwiclient needs to be available in the Python path
RUN git clone https://github.com/jks-prv/kiwiclient.git /app/kiwiclient \
    && cd /app/kiwiclient \
    && pip install --no-cache-dir numpy

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Copy and build Next.js dashboard
COPY src/dashboard/geographic-dashboard/ ./src/dashboard/geographic-dashboard/
WORKDIR /app/src/dashboard/geographic-dashboard
RUN npm install && npm run build

WORKDIR /app

# Create necessary directories
RUN mkdir -p /data/{recordings,cache,logs}

# Set Python path (include kiwiclient directory)
ENV PYTHONPATH=/app:/app/kiwiclient:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Health check (detects process type from command)
# For scheduler/worker: checks health file or Redis
# For API/dashboard: checks HTTP endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD if ps aux | grep -v grep | grep -q 'src.collectors.scheduler'; then python scripts/healthcheck.py scheduler; \
      elif ps aux | grep -v grep | grep -q 'src.collectors.worker'; then python scripts/healthcheck.py worker; \
      elif ps aux | grep -v grep | grep -q 'geographic_api'; then python scripts/healthcheck.py api; \
      elif ps aux | grep -v grep | grep -q 'qa_sample_api'; then python scripts/healthcheck.py qa_api; \
      elif ps aux | grep -v grep | grep -q 'npm start'; then python scripts/healthcheck.py dashboard; \
      else echo "Unknown process" && exit 1; fi

# Default command (overridden by fly.toml processes)
CMD ["python", "-m", "src.collectors.scheduler"]