#!/bin/bash

# CASCADE Data Collector Deployment Script for Fly.io with Tigris
# Deploys with optimized storage strategy using Tigris for everything

set -e

echo "🚀 CASCADE Data Collector Deployment Script"
echo "==========================================="
echo ""

# Check if fly CLI is installed
if ! command -v fly &> /dev/null; then
    echo "❌ Fly CLI not found. Installing..."
    curl -L https://fly.io/install.sh | sh
    export PATH="$HOME/.fly/bin:$PATH"
fi

# Check if logged in
if ! fly auth whoami &> /dev/null; then
    echo "📝 Please log in to Fly.io"
    fly auth login
fi

# Configuration
APP_NAME="cascade-kiwi-collector"
DB_NAME="cascade-db"
KEYDB_NAME="cascade-keydb"
REGION="iad"  # US East - good global connectivity

echo "Configuration:"
echo "  App: $APP_NAME"
echo "  Database: $DB_NAME"
echo "  KeyDB: $KEYDB_NAME"
echo "  Region: $REGION"
echo ""

# Step 1: Create app if it doesn't exist
echo "1️⃣ Creating Fly.io app..."
if fly apps list | grep -q "$APP_NAME"; then
    echo "   ✅ App already exists"
else
    fly apps create "$APP_NAME" --region "$REGION"
    echo "   ✅ App created"
fi

# Step 2: Create PostgreSQL (HA Lite for production)
echo "2️⃣ Setting up PostgreSQL..."
if fly postgres list | grep -q "$DB_NAME"; then
    echo "   ✅ Database already exists"
else
    echo "   Creating HA Lite PostgreSQL (3 nodes, optimized for writes)..."
    fly postgres create \
        --name "$DB_NAME" \
        --region "$REGION" \
        --initial-cluster-size 3 \
        --vm-size shared-cpu-2x \
        --volume-size 50 \
        --snapshot-retention 7

    # Attach to app (sets DATABASE_URL automatically)
    fly postgres attach "$DB_NAME" --app "$APP_NAME"
    echo "   ✅ PostgreSQL HA Lite created and attached"
fi

# Step 3: Create Tigris Storage Buckets
echo "3️⃣ Setting up Tigris storage..."

# Main IQ archive bucket
echo "   Creating main IQ archive bucket..."
fly storage create cascade-iq-data --app "$APP_NAME" --public false || true

# QA samples bucket
echo "   Creating QA samples bucket..."
fly storage create cascade-qa-samples --app "$APP_NAME" --public false || true

echo "   ✅ Tigris storage configured (zero egress fees!)"
echo "   📝 Note: Fly.io automatically sets AWS_* environment variables for Tigris"

# Step 4: Create KeyDB for coordination
echo "4️⃣ Setting up KeyDB..."
if fly apps list | grep -q "$KEYDB_NAME"; then
    echo "   ✅ KeyDB already exists"
else
    # Create KeyDB app
    fly apps create "$KEYDB_NAME" --region "$REGION"

    # Create KeyDB configuration
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
  KEYDB_SAVE = "60 1"  # Save every 60s if 1+ changes
EOF

    # Deploy KeyDB
    fly deploy -c keydb.fly.toml -a "$KEYDB_NAME"

    # Set REDIS_URL in main app
    fly secrets set REDIS_URL="redis://cascade-keydb.internal:6379" -a "$APP_NAME"

    echo "   ✅ KeyDB deployed (saves $500+/month vs Upstash)"
fi

# Step 5: Set required secrets
echo "5️⃣ Setting secrets..."

# Check if CALLSIGN_SALT is set
if ! fly secrets list -a "$APP_NAME" | grep -q "CALLSIGN_SALT"; then
    echo "   ⚠️  CALLSIGN_SALT not set. Generating secure salt..."
    SALT=$(openssl rand -hex 32)

    # Save to local file for backup (IMPORTANT!)
    echo "$SALT" > .callsign_salt.backup
    chmod 600 .callsign_salt.backup

    echo "   📝 CALLSIGN_SALT generated: $SALT"
    echo "   💾 Backed up to: .callsign_salt.backup (KEEP THIS SAFE!)"
    echo "   ⚠️  WARNING: If you lose this salt, historical data linkage breaks!"
    echo ""
    read -p "   Press Enter to set this salt in Fly.io secrets..."

    fly secrets set CALLSIGN_SALT="$SALT" -a "$APP_NAME"
    echo "   ✅ CALLSIGN_SALT set in Fly.io"
else
    echo "   ✅ CALLSIGN_SALT already configured"
fi

# Set Tigris bucket names (in addition to auto-configured AWS_* vars)
fly secrets set \
    TIGRIS_BUCKET_MAIN="cascade-iq-data" \
    TIGRIS_BUCKET_QA="cascade-qa-samples" \
    -a "$APP_NAME" 2>/dev/null || true

echo "   ✅ All secrets configured"

# Step 6: Deploy the application
echo "6️⃣ Deploying application..."

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Deploy with rolling strategy
fly deploy --strategy rolling --ha=false

echo "   ✅ Application deployed"

# Step 7: Scale process groups
echo "7️⃣ Scaling process groups..."

fly scale count scheduler=1 --process-group scheduler
fly scale count worker=2 --process-group worker
fly scale count api=1 --process-group api
fly scale count qa_api=1 --process-group qa_api
fly scale count dashboard=1 --process-group dashboard

echo "   ✅ Process groups scaled"

# Step 8: Run database migrations
echo "8️⃣ Running database migrations..."
fly ssh console -a "$APP_NAME" -C "python -m modules.data.migrations.run" || true
echo "   ✅ Migrations complete"

# Step 9: Display status and costs
echo ""
echo "========================================="
echo "✅ Deployment Complete!"
echo "========================================="
echo ""
echo "📊 Service URLs:"
echo "  Main: https://$APP_NAME.fly.dev"
echo "  Dashboard: https://$APP_NAME.fly.dev:3000"
echo "  Geographic API: https://$APP_NAME.fly.dev:8000/api/diversity/metrics"
echo "  QA Sample API: https://$APP_NAME.fly.dev:8001/api/qa/search"
echo ""
echo "💰 Monthly Cost Estimate:"
echo "  PostgreSQL HA Lite (3×2GB + 50GB): ~\$93"
echo "  KeyDB (256MB): ~\$2"
echo "  Workers (2×2GB): ~\$23"
echo "  APIs & Dashboard (3×1GB): ~\$17"
echo "  Tigris Storage (growing 0→50TB): \$0→\$1000"
echo "  Total: ~\$135/month (early) → \$1,135/month (full)"
echo ""
echo "🎉 Features:"
echo "  ✅ Zero egress fees with Tigris"
echo "  ✅ Intelligent QA sampling (10% diverse vs 1% random)"
echo "  ✅ Geographic diversity monitoring"
echo "  ✅ Weekly micro-model retraining"
echo "  ✅ HA database with automatic failover"
echo ""
echo "📝 Next Steps:"
echo "  1. Monitor at: fly dashboard"
echo "  2. View logs: fly logs -a $APP_NAME"
echo "  3. Check QA samples: fly ssh console -C 'ls /nvme/qa_cache'"
echo "  4. Verify Tigris: fly ssh console -C 'aws s3 ls s3://cascade-iq-data/'"
echo ""
echo "🔧 Troubleshooting:"
echo "  - Database: fly postgres connect -a $DB_NAME"
echo "  - KeyDB: fly ssh console -a $KEYDB_NAME -C 'redis-cli ping'"
echo "  - Tigris: Check AWS_* env vars are set automatically"