# CASCADE KiwiSDR Data Collector - Fly.io Deployment

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Fly.io                           │
├─────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │  Collector   │  │  Dashboard   │  │ Notifier │  │
│  │   Workers    │  │     Web      │  │  Worker  │  │
│  │  (Primary)   │  │   (1-2GB)    │  │  (256MB) │  │
│  │   (2-4GB)    │  │              │  │          │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬────┘  │
│         │                  │                │       │
│  ┌──────┴──────────────────┴────────────────┴────┐  │
│  │         Fly Postgres (Managed)                │  │
│  │              4GB Database                     │  │
│  └────────────────────────────────────────────────┘  │
│                           │                          │
└───────────────────────────┼──────────────────────────┘
                            │
                 ┌──────────┴──────────┐
                 │   Tigris Storage    │
                 │   10-15TB IQ Data   │
                 │   $20-30/month      │
                 └─────────────────────┘
```

## Fly.io Configuration Files

### fly.toml - Main Collector Application
```toml
# fly.toml
app = "cascade-collector"
primary_region = "ord"  # Chicago - central US location
kill_signal = "SIGINT"
kill_timeout = 30

[build]
  dockerfile = "Dockerfile.collector"

[env]
  # KiwiSDR configuration
  KIWI_TIMEOUT = "30"
  COLLECTION_MODE = "production"
  MAX_CONCURRENT_SDRS = "6"

  # Tigris configuration
  AWS_ENDPOINT_URL_S3 = "https://fly.storage.tigris.dev"
  BUCKET_NAME = "cascade-iq-data"

  # Collection parameters
  RECORDING_DURATION_SECONDS = "600"  # 10 minutes
  SAMPLE_RATE = "12000"
  CENTER_FREQUENCIES = "3576000,7080000,14080000,21080000,28080000,50303000"

[experimental]
  auto_rollback = true

[[services]]
  internal_port = 8080
  protocol = "tcp"
  auto_stop_machines = false  # Keep running for continuous collection
  auto_start_machines = true

  [[services.ports]]
    port = 80
    handlers = ["http"]
    force_https = true

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

  [[services.http_checks]]
    interval = "30s"
    grace_period = "10s"
    method = "GET"
    path = "/health"
    timeout = "5s"

[mounts]
  source = "cascade_data"
  destination = "/data"
  # Temporary storage for recordings before upload to Tigris

[[vm]]
  cpu_kind = "shared"
  cpus = 2
  memory_mb = 4096  # Need memory for concurrent SDR connections
```

### Dockerfile.collector
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libfftw3-dev \
    libsndfile1 \
    flac \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY modules/data/src/ ./src/
COPY modules/data/config/ ./config/

# Health check script
COPY health_check.py .

# Entry point
CMD ["python", "-m", "src.main"]
```

### requirements.txt
```txt
# KiwiSDR and audio processing
kiwiclient==0.2.1
numpy==1.24.3
scipy==1.11.4
soundfile==0.12.1

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
alembic==1.12.1

# AWS S3 (Tigris compatible)
boto3==1.29.7
aiobotocore==2.7.0  # Async S3 uploads

# FT8/WSPR decoding
pyFT8==0.1.0  # You may need to build this

# Monitoring and notifications
requests==2.31.0
schedule==1.2.0
python-dotenv==1.0.0

# Dashboard
fastapi==0.104.1
uvicorn==0.24.0
jinja2==3.1.2

# Utilities
pydantic==2.5.0
pyyaml==6.0.1
structlog==23.2.0
```

## Tigris Object Storage Integration

### tigris_storage.py
```python
import os
import asyncio
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import hashlib
import structlog

logger = structlog.get_logger()

class TigrisStorage:
    """Handle IQ data storage in Tigris object storage"""

    def __init__(self):
        # Tigris uses S3-compatible API
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.environ.get('AWS_ENDPOINT_URL_S3'),
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name='auto'
        )
        self.bucket_name = os.environ.get('BUCKET_NAME', 'cascade-iq-data')

    def upload_recording(self, local_path, metadata):
        """Upload IQ recording to Tigris with metadata"""

        # Generate S3 key with hierarchy
        # Format: year/month/day/band/hour/session_id.flac
        timestamp = metadata['start_time']
        s3_key = (
            f"{timestamp.year:04d}/"
            f"{timestamp.month:02d}/"
            f"{timestamp.day:02d}/"
            f"{metadata['band_name']}/"
            f"{timestamp.hour:02d}/"
            f"{metadata['session_id']}.flac"
        )

        # Calculate checksum
        with open(local_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # S3 metadata (searchable in Tigris)
        s3_metadata = {
            'session_id': metadata['session_id'],
            'kiwisdr_id': metadata['kiwisdr_id'],
            'center_frequency_hz': str(metadata['center_frequency_hz']),
            'duration_seconds': str(metadata['duration_seconds']),
            'file_hash': file_hash,
            'sdr_name': metadata.get('sdr_name', ''),
            'grid_square': metadata.get('grid_square', ''),
            'signal_count': str(metadata.get('signal_count', 0)),
            'k_index': str(metadata.get('k_index', 0)),
            'solar_flux': str(metadata.get('solar_flux', 0))
        }

        try:
            # Upload directly to archive tier for cost savings
            response = self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'Metadata': s3_metadata,
                    'StorageClass': 'GLACIER',  # Direct to archive tier
                    'ContentType': 'audio/flac'
                }
            )

            # Get the public URL (if needed)
            url = f"https://{self.bucket_name}.fly.storage.tigris.dev/{s3_key}"

            logger.info("Recording uploaded to Tigris",
                       s3_key=s3_key,
                       size_mb=os.path.getsize(local_path) / 1e6,
                       url=url)

            return {
                's3_key': s3_key,
                'url': url,
                'file_hash': file_hash,
                'size_bytes': os.path.getsize(local_path)
            }

        except ClientError as e:
            logger.error("Failed to upload to Tigris", error=str(e))
            raise

    def get_signed_url(self, s3_key, expiration=3600):
        """Generate presigned URL for temporary access"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error("Failed to generate presigned URL", error=str(e))
            return None

    def list_recordings(self, prefix='', max_items=1000):
        """List recordings with prefix filter"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_items
            )

            return [{
                'key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified'],
                'storage_class': obj.get('StorageClass', 'STANDARD')
            } for obj in response.get('Contents', [])]

        except ClientError as e:
            logger.error("Failed to list objects", error=str(e))
            return []

    def create_bucket_lifecycle(self):
        """Setup Tigris lifecycle rules for cost optimization"""
        lifecycle_config = {
            'Rules': [
                {
                    'ID': 'DeleteTempFiles',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'temp/'},
                    'Expiration': {'Days': 7}
                },
                # No transition rule needed - uploading directly to GLACIER
            ]
        }

        try:
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name,
                LifecycleConfiguration=lifecycle_config
            )
            logger.info("Lifecycle rules configured")
        except ClientError as e:
            logger.error("Failed to set lifecycle rules", error=str(e))
```

## Fly Postgres Setup

### Database initialization script
```bash
#!/bin/bash
# setup_fly_postgres.sh

# Create Fly Postgres cluster
flyctl postgres create \
  --name cascade-db \
  --region ord \
  --initial-cluster-size 1 \
  --vm-size shared-cpu-1x \
  --volume-size 10  # 10GB for metadata

# Attach to app
flyctl postgres attach cascade-db --app cascade-collector

# Get connection string
flyctl postgres connect -a cascade-db

# Initialize schema
flyctl postgres connect -a cascade-db < schema.sql

# Create database views
flyctl postgres connect -a cascade-db < dashboard_views.sql
```

## Deployment Commands

### Initial Setup
```bash
# 1. Install Fly CLI
curl -L https://fly.io/install.sh | sh

# 2. Login to Fly.io
flyctl auth login

# 3. Create Tigris bucket
flyctl storage create \
  --name cascade-iq-data \
  --public \
  --region ord

# 4. Create apps
flyctl apps create cascade-collector
flyctl apps create cascade-dashboard

# 5. Set up Postgres
./setup_fly_postgres.sh

# 6. Set secrets
flyctl secrets set \
  AWS_ACCESS_KEY_ID=tid_xxx \
  AWS_SECRET_ACCESS_KEY=tsec_xxx \
  DATABASE_URL=postgres://... \
  GMAIL_PASSWORD=xxxx-xxxx-xxxx-xxxx \
  --app cascade-collector

# 7. Deploy
flyctl deploy --app cascade-collector
flyctl deploy --app cascade-dashboard
```

### Monitoring & Logs
```bash
# View logs
flyctl logs --app cascade-collector

# SSH into container
flyctl ssh console --app cascade-collector

# Monitor metrics
flyctl monitor --app cascade-collector

# Check Tigris storage usage
flyctl storage dashboard cascade-iq-data
```

## Cost Estimate (Monthly)

| Service | Specs | Cost |
|---------|-------|------|
| **Fly.io Machines** | | |
| Collector (2 CPU, 4GB) | Always on | $35 |
| Dashboard (1 CPU, 2GB) | Always on | $17 |
| Notifier (shared, 256MB) | Always on | $3 |
| **Fly Postgres** | | |
| Development (1GB RAM, 10GB) | Managed | $15 |
| **Tigris Archive Storage** | | |
| Archive Storage (0-5TB) | $0.004/GB | $20 |
| Archive Storage (5-10TB) | $0.004/GB | $20 |
| Archive Storage (10-15TB) | $0.004/GB | $20 |
| Retrieval (if needed) | $0.01/GB | ~$5 |
| Bandwidth (minimal egress) | Minimal | $5 |
| **Total** | | **~$140/month** |

### Storage Cost Breakdown with Direct Archive

Uploading directly to Tigris archive tier saves significant costs:

| Storage Amount | Standard Cost | Archive Cost | Savings |
|----------------|--------------|--------------|---------|
| 5 TB | $100/mo | $20/mo | **80%** |
| 10 TB | $200/mo | $40/mo | **80%** |
| 15 TB | $300/mo | $60/mo | **80%** |

**Note**: Archive retrieval costs $0.01/GB if you need to access the data for training. Plan to retrieve in batches during model training phases.

## Scaling Strategy

### Phase 1: Initial Collection (Months 1-6)
- 1 collector instance
- 5TB Tigris storage
- ~$150/month

### Phase 2: Full Collection (Months 7-12)
- 2 collector instances (different regions)
- 10TB storage
- ~$200/month

### Phase 3: Storm Mode (As needed)
- Scale to 4+ collectors
- Distributed across regions
- Auto-scale based on K-index

## Advantages of Fly.io + Tigris

1. **Global Edge Network**: Collectors can run near KiwiSDRs
2. **S3 Compatibility**: Use existing S3 tools/libraries
3. **Auto-scaling**: Scale collectors during space weather events
4. **Built-in Monitoring**: Metrics, logs, alerts included
5. **Cost Effective**: ~50% cheaper than AWS for this workload
6. **Simple Deployment**: One command deploys everything

## Regional Deployment for Low Latency

```toml
# Deploy collectors near KiwiSDR concentrations
[regions]
  ord = 2  # Chicago - North America
  fra = 1  # Frankfurt - Europe
  nrt = 1  # Tokyo - Asia Pacific
  syd = 1  # Sydney - Oceania
```

This gives you global coverage with <100ms latency to most KiwiSDRs!