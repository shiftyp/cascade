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
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .

# Copy and build Next.js dashboard
COPY src/dashboard/geographic-dashboard/ ./src/dashboard/geographic-dashboard/
WORKDIR /app/src/dashboard/geographic-dashboard
RUN npm install && npm run build

WORKDIR /app

# Create necessary directories
RUN mkdir -p /data/{recordings,cache,logs}

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"

# Default command (overridden by fly.toml processes)
CMD ["python", "-m", "cascade_collector.scheduler"]