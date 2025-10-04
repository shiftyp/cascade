# CASCADE Data Collector - Deployment Guide

## Overview

The CASCADE Data Collector includes a Geographic Diversity Monitoring Dashboard to ensure balanced global coverage of HF radio propagation data. This guide covers deployment to Fly.io and local development setup.

## Features

- **Geographic Diversity Dashboard**: Real-time monitoring of collection coverage
- **Automatic Rebalancing**: Dynamic adjustment of collection priorities
- **Southern Hemisphere Priority**: 3x weight for underrepresented regions
- **Global SDR Network**: Connects to 50-100+ KiwiSDR/WebSDR receivers worldwide
- **QA Waterfall Viewer**: Visualize 1% QA samples with interactive waterfall displays
- **Simpson's Diversity Index**: Statistical metrics for geographic distribution
- **Latitude Band Quotas**: Ensure coverage across Arctic, Temperate, Tropical, and Antarctic zones
- **Reciprocal Path Inference**: Generate synthetic southern hemisphere data from northern transmissions

## Local Development

### Prerequisites

- Docker and Docker Compose
- Node.js 20+ (for dashboard development)
- Python 3.11+

### Quick Start

1. **Start all services**:
```bash
docker-compose up -d
```

2. **Access services**:
- Dashboard: http://localhost:3000
- Geographic API: http://localhost:8000/api/diversity/metrics
- QA Sample API: http://localhost:8001/api/qa/search
- Waterfall API: http://localhost:8001/api/waterfall
- PostgreSQL Admin: http://localhost:8082
- Redis Commander: http://localhost:8081

3. **View logs**:
```bash
docker-compose logs -f scheduler
docker-compose logs -f diversity-api
```

### Development Workflow

1. **Database migrations**:
```bash
docker-compose exec scheduler python -m modules.data.migrations.run
```

2. **Add test data**:
```bash
curl -X POST http://localhost:8000/api/diversity/add-sample \
  -H "Content-Type: application/json" \
  -d '{"grid_square": "FN42", "hours": 10, "is_ocean_path": false}'
```

3. **Scale workers**:
```bash
docker-compose up -d --scale worker=4
```

## Production Deployment (Fly.io)

### Prerequisites

1. **Install Fly CLI**:
```bash
curl -L https://fly.io/install.sh | sh
```

2. **Login to Fly.io**:
```bash
fly auth login
```

### Required Secrets

Fly.io **automatically configures** most secrets when you provision services:

#### Auto-Configured (No Manual Setup Required)
- ✅ **DATABASE_URL** - Set automatically by `fly postgres attach`
- ✅ **AWS_ACCESS_KEY_ID** - Set automatically by `fly storage create` (Tigris)
- ✅ **AWS_SECRET_ACCESS_KEY** - Set automatically by `fly storage create` (Tigris)
- ✅ **AWS_ENDPOINT_URL_S3** - Set automatically by `fly storage create` (Tigris)
- ✅ **BUCKET_NAME** - Set automatically by `fly storage create` (Tigris)
- ✅ **REDIS_URL** - Set automatically by Fly.io Redis addon (if used)

#### Manual Configuration Required
- ⚠️ **CALLSIGN_SALT** - Generate with `openssl rand -hex 32` and set manually
- 🔧 **GMAIL_APP_PASSWORD** - Optional, for email notifications
- 🔧 **GMAIL_SENDER_EMAIL** - Optional, for email notifications
- 🔧 **NOTIFICATION_RECIPIENTS** - Optional, comma-separated email list

### Automated Deployment

Run the deployment script:
```bash
./deploy.sh
```

This will:
- Create PostgreSQL HA cluster (3-node, 250GB) → **Sets DATABASE_URL automatically**
- Create Tigris storage bucket → **Sets AWS credentials automatically**
- Prompt for CALLSIGN_SALT (only required manual secret)
- Deploy to US East (IAD) region
- Configure auto-scaling workers
- Set up automated backups

### Manual Deployment

1. **Create app**:
```bash
fly apps create cascade-kiwi-collector --region iad
```

2. **Create PostgreSQL** (Start Small, Scale Up):

Choose a configuration based on your phase:

**Development/Testing** (~$8/month):
```bash
fly postgres create \
  --name cascade-db \
  --region iad \
  --initial-cluster-size 1 \
  --vm-size shared-cpu-1x \
  --volume-size 10

fly postgres attach cascade-db --app cascade-kiwi-collector
```

**Production Starter** (~$15/month) - Single node for testing:
```bash
fly postgres create \
  --name cascade-db \
  --region iad \
  --initial-cluster-size 1 \
  --vm-size shared-cpu-2x \
  --volume-size 50 \
  --snapshot-retention 7

fly postgres attach cascade-db --app cascade-kiwi-collector
```

**Production HA Lite** (~$93/month) - **RECOMMENDED for write-heavy workload**:
```bash
# 3-node HA cluster with smaller CPUs (sufficient for writes)
fly postgres create \
  --name cascade-db \
  --region iad \
  --initial-cluster-size 3 \
  --vm-size shared-cpu-2x \
  --volume-size 50 \
  --snapshot-retention 7

fly postgres attach cascade-db --app cascade-kiwi-collector
```

**Production HA Standard** (~$177-241/month) - For high query load:
```bash
# 3-node HA cluster (migrate to this when starting 24/7 collection)
fly postgres create \
  --name cascade-db \
  --region iad \
  --initial-cluster-size 3 \
  --vm-size shared-cpu-4x \
  --volume-size 100 \
  --snapshot-retention 7

fly postgres attach cascade-db --app cascade-kiwi-collector
```

**Migration Path** (Start cheap with HA, scale storage/CPU as needed):

1. **Month 0-1 (Development)**: Single node 50GB (~$15/month) - Testing only
2. **Month 1+ (Production)**: **HA Lite 50GB (~$93/month)** - Recommended start
3. **Month 6+**: Extend volumes to 100GB (~$110/month) as data grows
4. **Month 12+**: Extend to 250GB (~$143/month) or upgrade to shared-cpu-4x if queries slow
5. **Long-term**: Archive old data to S3 to keep DB at 100GB

**Why HA Lite?**
- ✅ **HA protection** from day 1 (no data loss risk)
- ✅ **Write-optimized**: shared-cpu-2x handles INSERT-heavy workload fine
- ✅ **Saves $84/month** vs HA Standard ($93 vs $177)
- ✅ **Easy upgrade path**: Scale CPU/storage independently later
- ✅ **12-month savings**: $1,008 vs starting with HA Standard

**Database Growth Projections**:
- **Month 1**: ~1-2 GB (testing)
- **Month 3**: ~10-20 GB (initial collection)
- **Month 6**: ~70 GB (active collection)
- **Month 12**: ~140 GB (halfway)
- **Month 18**: ~250 GB (full 250K hours)

**How to Migrate/Scale Later**:

```bash
# Option 1: Extend existing volumes (single node)
fly volumes list -a cascade-db
fly volumes extend <volume-id> --size 100 -a cascade-db

# Option 2: Migrate to HA cluster
# 1. Create new HA cluster
fly postgres create --name cascade-db-ha --initial-cluster-size 3 --vm-size shared-cpu-4x --volume-size 100
# 2. Backup old database
fly postgres connect -a cascade-db
pg_dump cascade > /tmp/backup.sql
# 3. Restore to new cluster
fly postgres connect -a cascade-db-ha
psql cascade < /tmp/backup.sql
# 4. Update app to use new cluster
fly postgres detach cascade-db --app cascade-kiwi-collector
fly postgres attach cascade-db-ha --app cascade-kiwi-collector

# Option 3: Scale up machine size
fly scale vm shared-cpu-4x --memory 8GB -a cascade-db
```

**Cost Savings by Starting Small**:
- Development phase (3 months): Save $233/month vs full HA
- Total savings months 0-3: **$699**
- Then scale to HA when collection starts

3. **Create Tigris Storage** (Automatically provisioned by Fly.io):

```bash
# Tigris is auto-configured when you deploy to Fly.io
# It will automatically set these secrets:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_ENDPOINT_URL_S3 (https://fly.storage.tigris.dev)
# - BUCKET_NAME

# To manually create bucket (if not auto-created):
fly storage create --name cascade-iq-data --public false

# This provides:
# - $0.02/GB/month storage (50% cheaper than AWS S3 Standard)
# - Zero egress fees (vs $0.09/GB on AWS)
# - S3-compatible API
# - Automatic Fly.io integration
```

4. **Create KeyDB for Distributed Coordination**:

**Cost Comparison**:

| Option | Monthly Cost | Commands/Month | Management |
|--------|--------------|----------------|------------|
| Upstash Free | $0 | 500K | ❌ Too limited (need 260M-520M) |
| Upstash Pay-as-you-go | $520-1,040 | Unlimited | ❌ Expensive for locks |
| **Self-Managed KeyDB** | **$2-4** | **Unlimited** | ✅ **RECOMMENDED** |

**RECOMMENDED: Self-Managed KeyDB** (~$2-4/month, saves $516-1,036/month!):

CASCADE uses Redis for ephemeral coordination only (locks, queues), not persistent data. With 50-100 workers doing ~2 lock operations per second, that's 260M-520M commands/month. Upstash would cost $520-1,040/month for this. Self-managed KeyDB costs $2-4/month.

```bash
# Create separate KeyDB app
fly apps create cascade-keydb --region iad

# Create fly.toml for KeyDB
cat > keydb.fly.toml << 'EOF'
app = "cascade-keydb"
primary_region = "iad"

[build]
  image = "eqalpha/keydb:latest"

[[vm]]
  memory = '256mb'
  cpu_kind = 'shared'
  cpus = 1

[[services]]
  internal_port = 6379
  protocol = "tcp"

  [[services.ports]]
    port = 6379

[env]
  KEYDB_THREADS = "4"
EOF

# Deploy KeyDB
fly deploy -c keydb.fly.toml -a cascade-keydb

# KeyDB will be available at: cascade-keydb.internal:6379
# Workers auto-discover via Fly internal DNS (no REDIS_URL secret needed)
# Or set explicitly: fly secrets set REDIS_URL="redis://cascade-keydb.internal:6379" -a cascade-kiwi-collector
```

**Why Self-Managed KeyDB?**
- ✅ **130-260x cheaper**: $2-4/month vs $520-1,040/month
- ✅ **Unlimited commands**: No pay-per-operation costs
- ✅ **No backup needed**: Ephemeral locks/queues (rebuilds from Postgres on restart)
- ✅ **KeyDB = multi-threaded**: 4 threads on shared-cpu-1x for better performance
- ✅ **Simple management**: Just monitor uptime, no complex operations
- ✅ **Annual savings**: $6,192-12,432 vs Upstash

**What CASCADE Uses Redis For**:
- Distributed locks (prevent multiple workers claiming same SDR)
- Work queue (SDR assignment distribution)
- Worker health tracking (30-second heartbeats)
- Ephemeral coordination state

**Note**: All persistent data goes to PostgreSQL. Redis failure = workers restart and reacquire locks. No data loss.

4. **Set secrets**:
```bash
fly secrets set \
  TIGRIS_ACCESS_KEY="tid_xxx" \
  TIGRIS_SECRET_KEY="tsec_xxx" \
  TIGRIS_BUCKET="cascade-iq-data" \
  NOAA_API_KEY="optional" \
  --app cascade-kiwi-collector

# DATABASE_URL is automatically set by `fly postgres attach`
# REDIS_URL should be set separately if using external service
```

5. **Deploy**:
```bash
fly deploy --strategy rolling --ha=false
```

6. **Scale process groups**:
```bash
# Scale individual process groups
fly scale count scheduler=1 --process-group scheduler
fly scale count worker=2 --process-group worker
fly scale count api=1 --process-group api
fly scale count dashboard=1 --process-group dashboard
```

7. **Setup autoscaling for workers** (optional):
```bash
# Autoscaling is handled by min_machines_running in fly.toml
# Workers will auto-start/stop based on load

# For metrics-based autoscaling, use the autoscaler app:
# See: https://fly.io/docs/reference/autoscaling/
```

8. **Verify deployment**:
```bash
# Check deployment status
fly status

# View logs
fly logs

# Monitor metrics
fly dashboard
```

**Note on Regions**: Geographic diversity is achieved by connecting to globally distributed KiwiSDR/WebSDR receivers, not by deploying the collector to multiple regions. A single region deployment (US East - iad) is sufficient and simpler to manage.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|------------|---------|
| `DIVERSITY_MONITORING` | Enable geographic diversity monitoring | `enabled` |
| `SCARCE_REGION_SLOTS` | Percentage reserved for underrepresented regions | `0.2` (20%) |
| `DIVERSITY_HOUR_UTC` | Hour (UTC) for daily diversity collection | `14` |
| `PREFER_SCARCE_REGIONS` | Prioritize scarce regions | `true` |
| `MIN_SDRS` | Minimum concurrent SDRs | `6` |
| `MAX_SDRS` | Maximum concurrent SDRs during events | `50` |

### Geographic Quotas

Default latitude band quotas (minimum 20% each):
- Arctic (>66.5°N)
- Temperate (23.5-66.5°)
- Tropical (±23.5°)
- Antarctic (<-66.5°S)

Hemispheric balance targets:
- Northern: 40%
- Southern: 40%
- Equatorial: 20%

Ocean path minimum: 30%

## Database Operations

### Automated Backups

Set up automated backups to Tigris S3:

```bash
# Create backup script (run via cron on scheduler machine)
cat > /app/scripts/backup_postgres.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump $DATABASE_URL | gzip | aws s3 cp - s3://$TIGRIS_BUCKET/backups/postgres_$DATE.sql.gz
# Cleanup old backups (keep last 30 days)
aws s3 ls s3://$TIGRIS_BUCKET/backups/ | awk '{print $4}' | sort | head -n -30 | xargs -I {} aws s3 rm s3://$TIGRIS_BUCKET/backups/{}
EOF

# Schedule daily backups at 2 AM UTC
fly ssh console -C "echo '0 2 * * * /app/scripts/backup_postgres.sh' | crontab -"
```

### Manual Backup & Restore

```bash
# Create manual backup
fly postgres connect -a cascade-db
pg_dump cascade > backup_$(date +%Y%m%d).sql
\q

# Restore from backup
fly postgres connect -a cascade-db
psql cascade < backup_20250930.sql
```

### Database Monitoring

```bash
# Check database size
fly ssh console -a cascade-db -C "psql -c \"SELECT pg_size_pretty(pg_database_size('cascade'));\""

# Check table sizes
fly ssh console -a cascade-db -C "psql -c \"
  SELECT schemaname, tablename,
         pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
  FROM pg_tables
  WHERE schemaname = 'public'
  ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
  LIMIT 10;
\""

# Check connection count
fly ssh console -a cascade-db -C "psql -c \"SELECT count(*) FROM pg_stat_activity;\""

# Check replication status
fly postgres list -a cascade-db
```

### Disk Space Management

When approaching storage limits:

```bash
# Check current usage
fly volumes list -a cascade-db

# Extend volumes (all replicas)
fly volumes extend vol_xxx --size 500 -a cascade-db
fly volumes extend vol_yyy --size 500 -a cascade-db
fly volumes extend vol_zzz --size 500 -a cascade-db

# Archive old data to S3
# Export propagation_records older than 12 months
fly ssh console -a cascade-db -C "
  psql -c \"COPY (SELECT * FROM propagation_records WHERE created_at < NOW() - INTERVAL '12 months') TO STDOUT CSV HEADER\" | \
  gzip | aws s3 cp - s3://cascade-iq-data/archive/propagation_records_archive.csv.gz
"

# Delete archived records
fly ssh console -a cascade-db -C "
  psql -c \"DELETE FROM propagation_records WHERE created_at < NOW() - INTERVAL '12 months';\"
  psql -c \"VACUUM FULL propagation_records;\"
"
```

### Performance Tuning

For 600M+ row tables, optimize Postgres configuration:

```bash
# Edit postgresql.conf via Fly console
fly ssh console -a cascade-db

# Recommended settings for 8GB RAM:
# shared_buffers = 2GB
# effective_cache_size = 6GB
# work_mem = 16MB
# maintenance_work_mem = 512MB
# max_connections = 100
# random_page_cost = 1.1 (for SSD)

# Apply changes
pg_ctl reload
```

## Monitoring

### Dashboard Access

Production URLs:
- Main: https://cascade-kiwi-collector.fly.dev
- Dashboard: https://cascade-kiwi-collector.fly.dev:3000
- API: https://cascade-kiwi-collector.fly.dev:8000/api/diversity/metrics

### Key Metrics

1. **Diversity Score**: Overall geographic diversity (target: >0.7)
2. **Simpson's Index**: Distribution uniformity (target: >0.7)
3. **Hemisphere Balance**: Ratio between hemispheres (target: 0.8-1.2)
4. **Continental Coverage**: Number of continents (target: 7/7)

### Alerts

Configure alerts for:
- Diversity score < 0.5
- Any region < 50% of quota
- Hemisphere imbalance > 2:1
- Worker CPU > 80%
- **Database size > 80%** of volume capacity
- **Postgres replication lag** > 60 seconds
- **Failed backup** jobs

### Commands

```bash
# View logs
fly logs -a cascade-kiwi-collector

# Check status
fly status -a cascade-kiwi-collector

# View metrics
fly dashboard --metrics

# SSH into container
fly ssh console -a cascade-kiwi-collector

# Scale workers
fly scale count worker=N -a cascade-kiwi-collector
```

## Troubleshooting

### Low Diversity Score

1. Check underrepresented regions:
```bash
curl https://cascade-kiwi-collector.fly.dev:8000/api/diversity/warnings
```

2. View recommendations:
```bash
curl https://cascade-kiwi-collector.fly.dev:8000/api/diversity/recommendations
```

3. Manually trigger southern collection:
```bash
fly ssh console -c "python -m modules.data.src.collectors.southern_priority"
```

### Database Issues

1. Check connection:
```bash
fly postgres connect -a cascade-db
```

2. Run migrations:
```bash
fly ssh console -c "python -m modules.data.migrations.run"
```

### Dashboard Not Loading

1. Check Next.js build:
```bash
fly ssh console -c "cd /app/src/dashboard/geographic-dashboard && npm run build"
```

2. Verify API connection:
```bash
curl http://localhost:8000/api/diversity/metrics
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  Global KiwiSDR/WebSDR Network             │
│  (50-100 baseline, scales to 200+ during events)          │
│  (800-1100 total receivers for rotation)                   │
└─────────────────────┬──────────────────────────────────────┘
                      │ Connects to
┌─────────────────────▼──────────────────────────────────────┐
│              Fly.io Region: US East (IAD)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │              CASCADE Data Collector                │    │
│  ├───────────────────────────────────────────────────┤    │
│  │                                                     │    │
│  │  ┌──────────────┐  ┌────────────────────────┐    │    │
│  │  │  Scheduler   │  │   Diversity Dashboard   │    │    │
│  │  │  (Singleton) │  │   (Next.js + FastAPI)   │    │    │
│  │  └──────────────┘  └────────────────────────┘    │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────┐    │    │
│  │  │         Worker Pool (2-10)                │    │    │
│  │  │  Auto-scales with space weather events    │    │    │
│  │  │  2-3: Summer quiet | 7-10: Winter storms │    │    │
│  │  └──────────────────────────────────────────┘    │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐        │
│  │PostgreSQL│  │  Redis/  │  │  Tigris S3       │        │
│  │(Fly HA)  │  │  KeyDB   │  │  (35-75TB)       │        │
│  │~250GB    │  │          │  │  200K-300K hrs   │        │
│  └──────────┘  └──────────┘  └──────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principles**:
- **Single-region deployment**: Simplified operations, lower cost
- **Geographic diversity at source**: Connects to globally distributed SDRs
- **Auto-scaling workers**: 2-10 workers based on number of active SDR connections
- **Centralized monitoring**: Dashboard tracks coverage across all collected data

## Infrastructure Costs

### Phase-Based Cost Strategy

**Development Phase (Month 1)**:
| Component | Configuration | Monthly Cost |
|-----------|--------------|--------------|
| **Postgres Dev** | 1× shared-cpu-2x (4GB) + 50GB volume | $28.90 |
| **KeyDB** | 1× shared-cpu-1x (256MB) | $1.94 |
| **Scheduler** | 1× shared-cpu-2x (2GB) | $11.39 |
| **Workers** | 2× shared-cpu-2x (2GB) | $22.78 |
| **API Services** | 2× shared-cpu-1x (1GB) | $11.40 |
| **Dashboard** | 1× shared-cpu-1x (1GB) | $5.70 |
| **Tigris S3** | 0.5TB @ $0.02/GB (zero egress fees) | $10.00 |
| **Total** | | **~$92/month** |

**Production Phase (Months 2-18)** - Using HA Lite + KeyDB:
| Component | Configuration | Monthly Cost Range |
|-----------|--------------|-------------------|
| **Postgres HA Lite** | 3× shared-cpu-2x (2GB) + 3× 50-100GB volumes | $93-110 |
| **KeyDB** | 1× shared-cpu-1x (256MB) | $1.94 |
| **Scheduler** | 1× shared-cpu-2x (2GB) | $11.39 |
| **Workers** | 2-10× shared-cpu-2x (2GB), scales with events | $23-114 |
| **API Services** | 2× shared-cpu-1x (1GB) | $11.40 |
| **Dashboard** | 1× shared-cpu-1x (1GB) | $5.70 |
| **Tigris S3** | Grows 2TB → 46TB @ $0.02/GB (zero egress!) | $40-920 |
| **Total (early)** | Months 2-6 | **$192-436/month** |
| **Total (mid)** | Months 7-11 | **$464-774/month** |
| **Total (late)** | Months 12-18 | **$774-1,070/month** |

**Cost Savings**:
- ✅ **HA Lite vs HA Standard**: Save $84/month ($1,008/year)
- ✅ **KeyDB vs Upstash**: Save $516-1,036/month ($6,192-12,432/year)
- ✅ **Combined savings**: $600-1,120/month ($7,200-13,440/year!)

**Note**: Extend Postgres volumes as data grows (~$11/month per 100GB added to 3-node cluster)

### Cost Optimization Strategies

1. **Archive old propagation records to S3 after 12 months**:
   - Keeps DB at ~100GB instead of 250GB
   - Reduces Postgres volumes to 100GB each
   - Saves: ~$67/month in volume costs
   - New DB cost: ~$174/month

2. **Use development config initially** (0-3 months):
   - Single node shared-cpu-2x (4GB) + 50GB volume
   - Cost: ~$29/month
   - Migrate to HA when collection starts

3. **Scale workers dynamically**:
   - 2 workers baseline: $23/month
   - 10 workers during events: $114/month
   - Average: ~$45/month

### First 6 Months Cost (With Event Weighting)

| Month | Phase | Postgres | KeyDB | Workers | Tigris | **Total** |
|-------|-------|----------|-------|---------|--------|-----------|
| 1 | Dev/Testing | $28.90 | $1.94 | $35 | $50 | **$192** |
| 2 | HA Lite + Winter | $93.00 | $1.94 | $40 | $104 | **$251** |
| 3 | Spring Equinox | $93.00 | $1.94 | $50 | $168 | **$377** |
| 4 | Post-Equinox | $93.00 | $1.94 | $42 | $216 | **$369** |
| 5 | Spring Declining | $93.00 | $1.94 | $38 | $260 | **$408** |
| 6 | Summer Low | $93.00 | $1.94 | $32 | $294 | **$436** |
| **Total** | | | | | | **$2,033** |

**First 6 months**: $2,033 (~81,000 hours = **$0.025/hour**)

### 18-Month Total Cost Estimate (With Seasonal & Event Weighting)

**Important**: These projections account for seasonal collection variance, geographic weighting overhead, and event-driven scaling. See [Weighted Collection Rate Model](../../docs/training/data_pipeline.md#weighted-collection-rate-model) for methodology.

#### Month-by-Month Cost Breakdown

| Month | Season | Expected Activity | Storage (TB) | Tigris Cost | Workers | Compute | **Total** |
|-------|--------|-------------------|--------------|-------------|---------|---------|-----------|
| **1** | Winter | Minor storms | 2.5 | $50 | 3-5 | $35 | **$192** |
| **2** | Winter | Moderate | 5.2 | $104 | 3-6 | $40 | **$251** |
| **3** | Spring | **Equinox peak** | 8.4 | $168 | 5-8 | $50 | **$377** |
| **4** | Spring | Post-equinox | 10.8 | $216 | 4-6 | $42 | **$369** |
| **5** | Spring | Declining | 13.0 | $260 | 3-5 | $38 | **$408** |
| **6** | Summer | Low activity | 14.7 | $294 | 2-4 | $32 | **$436** |
| **7** | Summer | **Quiet period** | 16.3 | $326 | 2-3 | $28 | **$464** |
| **8** | Summer | Sporadic-E | 18.1 | $362 | 3-5 | $35 | **$507** |
| **9** | Autumn | **Equinox peak** | 21.1 | $422 | 5-8 | $52 | **$624** |
| **10** | Autumn | Active | 23.6 | $472 | 4-6 | $45 | **$627** |
| **11** | Autumn | Rising | 26.1 | $522 | 4-7 | $48 | **$680** |
| **12** | Winter | **Storm season** | 29.1 | $582 | 5-8 | $52 | **$774** |
| **13** | Winter | Peak activity | 32.2 | $644 | 6-9 | $58 | **$852** |
| **14** | Winter | **Major events** | 35.8 | $716 | 7-10 | $65 | **$931** |
| **15** | Spring | **Equinox peak** | 39.0 | $780 | 6-9 | $60 | **$990** |
| **16** | Spring | Active | 41.9 | $838 | 5-8 | $54 | **$1,002** |
| **17** | Summer | Declining | 43.8 | $876 | 3-5 | $38 | **$1,024** |
| **18** | Summer | Final push | 45.9 | $918 | 4-6 | $42 | **$1,070** |
| | | | | | | | |
| **Total** | | | **45.9 TB** | | | | **$11,578** |

**Cost Variance Analysis**:
- **Best case** (quiet solar minimum, minimal events): **$9,800** (-15%)
- **Expected case** (moderate seasonal variance): **$11,578** (baseline)
- **Worst case** (multiple X-class flares + winter storms): **$14,200** (+23%)
- **Recommended budget**: **$13,500** (includes 17% contingency)

**Storage Growth Pattern**:
- **Summer months** (6-8, 17-18): 250-350 hrs/day → 8,000-11,000 hrs/month → 1.5-2.0 TB/month
- **Winter months** (1-2, 12-14): 450-600 hrs/day → 13,500-18,000 hrs/month → 2.5-3.3 TB/month
- **Equinox peaks** (3, 9, 15): 500-700 hrs/day → 15,000-21,000 hrs/month → 2.7-3.8 TB/month
- **Event bursts**: Up to 1,500-2,000 hrs/day during major storms (5-day duration typical)

**Worker Scaling Pattern**:
- **Baseline** (summer quiet): 2-3 workers × $11.39/month = $23-34/month
- **Moderate** (typical collection): 4-6 workers = $45-68/month
- **Peak events** (winter storms, equinoxes): 7-10 workers = $80-114/month
- **Extreme** (K=8+ storms, X-class flares): Up to 10 workers temporarily

**Key Cost Drivers by Phase**:

| Phase | Duration | Primary Cost | Secondary Cost | Variance |
|-------|----------|--------------|----------------|----------|
| **Dev/Testing** | Month 1 | Workers ($35) | Tigris ($50) | Low (±10%) |
| **Ramp-Up** | Months 2-5 | Tigris ($104-260) | Workers ($38-50) | Moderate (±15%) |
| **Mid Collection** | Months 6-11 | Tigris ($294-522) | Workers ($28-52) | High (±20%) |
| **Peak Activity** | Months 12-16 | Tigris ($582-838) | Workers ($52-60) | Very High (±25%) |
| **Final Phase** | Months 17-18 | Tigris ($876-918) | Workers ($38-42) | Low (±10%) |

**Cost per hour of collected data**:
- **Nominal**: $11,578 / 252,000 hours = **$0.046/hour**
- **With contingency**: $13,500 / 252,000 hours = **$0.054/hour**
- **Range**: $0.039-0.056/hour depending on event frequency

**Massive Cost Savings** (thanks to Tigris zero egress + KeyDB):
- ✅ **vs HA Standard + Upstash**: Save $5,000-10,000+ over 18 months
- ✅ **vs AWS S3 with egress**: Save $6,000-9,000 in transfer fees alone
- ✅ **Tigris zero egress**: Unlimited free downloads for weekly training (save $4,000+)
- ✅ **KeyDB vs Upstash**: Save $6,192-12,432 annually on coordination costs
- ✅ **HA Lite vs HA Standard**: Save $1,008 annually on database costs

### Alternative: Fly Managed Postgres

For comparison, using Fly MPG Launch plan:

| Component | Monthly Cost |
|-----------|--------------|
| Postgres MPG Launch (250GB) | $352.00 |
| Other services (same) | $883.00 |
| **Total** | **$1,235/month** |

**Difference**: +$110/month vs self-managed
**18-month difference**: +$1,980

**MPG Trade-offs:**
- ✅ 24/7 support, automatic backups
- ✅ Saves ~20 hours/month of DB operations
- ❌ Hard 1TB limit (can't grow beyond)
- ❌ More expensive at scale

### Recommendation

**Start with Production Starter ($29/month), scale to HA when needed**:

**Phase 1 (Months 0-3)**: Single node, 50GB
- Cost: $29/month
- Good for: Development, testing, initial collection
- Risk: No HA, but acceptable during development

**Phase 2 (Months 3-18)**: Migrate to 3-node HA, 100GB (extend to 250GB)
- Cost: $177/month (start) → $241/month (at 250GB)
- Good for: 24/7 production collection
- HA protects against data loss during critical collection

**Total 18-month cost**: $16,281
- vs Fly MPG: **$2,679 savings**
- vs starting with full HA: **$699 additional savings in dev phase**

**Critical Setup Tasks**:
1. ✅ Set up automated backups immediately (see Database Operations section)
2. ✅ Configure disk space alerts (>80% usage)
3. ✅ Test failover procedures before production
4. ✅ Document migration path to HA cluster

## Support

- Issues: https://github.com/cascade/data-collector/issues
- Documentation: https://cascade-docs.fly.dev
- Monitoring: https://fly-metrics.net/d/cascade-kiwi-collector

## License

Copyright (c) 2024 CASCADE Project. All rights reserved.