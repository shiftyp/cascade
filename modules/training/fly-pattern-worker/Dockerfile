FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy entire modules directory
COPY modules/ /app/modules/

# Install Python dependencies
COPY modules/training/fly-pattern-worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Make worker script executable
RUN chmod +x /app/modules/training/fly-pattern-worker/worker.py

CMD ["python", "/app/modules/training/fly-pattern-worker/worker.py"]
