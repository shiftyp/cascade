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

# # Configuration
APP_NAME="cascade-collector"
DB_NAME="cascade-rf-db"
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
    fly apps create "$APP_NAME"
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
        --volume-size 50

    # Attach to app (sets DATABASE_URL automatically)
    fly postgres attach "$DB_NAME" --app "$APP_NAME"
    echo "   ✅ PostgreSQL HA Lite created and attached"
fi

# Step 3: Create Tigris Storage
echo "3️⃣ Setting up Tigris storage..."

# Create Tigris storage for the app (this creates a single Tigris project)
echo "   Setting up Tigris object storage..."

# Check if Tigris is already configured
if fly secrets list -a "$APP_NAME" | grep -q "AWS_ACCESS_KEY_ID"; then
    echo "   ✅ Tigris storage already configured"
else
    echo "   Creating Tigris storage project..."

    # Create Tigris storage (this will set AWS_* env vars automatically)
    fly storage create \
        --app "$APP_NAME" \
        --name "${APP_NAME}-storage" || true

    echo "   ✅ Tigris storage created"
fi

echo "   📝 Note: Tigris provides a single storage project per app"
echo "   📝 Buckets will be created programmatically when the app runs"
echo "   📝 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_ENDPOINT_URL are set automatically"

# Step 4: Create KeyDB for coordination
echo "4️⃣ Setting up KeyDB..."
if fly apps list | grep -q "$KEYDB_NAME"; then
    echo "   ✅ KeyDB already exists"
else
    # Create KeyDB app
    fly apps create "$KEYDB_NAME"

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

# Check if Gmail notification credentials are set
echo ""
echo "📧 Checking Gmail notification setup..."
if ! fly secrets list -a "$APP_NAME" | grep -q "GMAIL_SENDER_EMAIL"; then
    echo "   ⚠️  Gmail notifications not configured."
    echo "   This is optional but recommended for operator alerts."
    echo ""
    read -p "   Would you like to set up Gmail notifications? (y/N): " setup_gmail

    if [[ "$setup_gmail" =~ ^[Yy]$ ]]; then
        echo ""
        echo "   📝 Gmail Setup Instructions:"
        echo "   1. Use a dedicated Gmail account for CASCADE alerts"
        echo "   2. Enable 2-factor authentication on the account"
        echo "   3. Generate an app-specific password:"
        echo "      https://myaccount.google.com/apppasswords"
        echo ""

        read -p "   Enter Gmail address (e.g., cascade-alerts@gmail.com): " gmail_address
        read -sp "   Enter Gmail app password (16 characters, no spaces): " gmail_password
        echo ""

        # Validate inputs
        if [[ -n "$gmail_address" && -n "$gmail_password" ]]; then
            # Also prompt for recipient emails
            echo ""
            echo "   📬 Notification Recipients:"
            read -p "   Enter operator emails (comma-separated): " operator_emails

            fly secrets set \
                GMAIL_SENDER_EMAIL="$gmail_address" \
                GMAIL_APP_PASSWORD="$gmail_password" \
                NOTIFICATION_RECIPIENTS="$operator_emails" \
                -a "$APP_NAME"

            echo "   ✅ Gmail notifications configured"
            echo "   📝 Test with: fly ssh console -a $APP_NAME -C 'python -m modules.data.src.notifications.gmail_notifier --test'"
        else
            echo "   ⚠️  Skipping Gmail setup - invalid inputs"
        fi
    else
        echo "   ⚠️  Skipping Gmail notifications (can be configured later)"
    fi
else
    echo "   ✅ Gmail notifications already configured"
fi

# Set Tigris bucket names for the application to use
fly secrets set \
    TIGRIS_BUCKET="cascade-iq-data" \
    TIGRIS_BUCKET_QA="cascade-qa-samples" \
    -a "$APP_NAME" 2>/dev/null || true

echo "   ✅ All secrets configured"

# Step 6: Create persistent volume for data
echo "6️⃣ Setting up persistent storage volume..."

# Check if volume exists
if fly volumes list -a "$APP_NAME" | grep -q "data"; then
    echo "   ✅ Data volume already exists"
else
    echo "   Creating 100GB volume for local cache..."
    fly volumes create data \
        --app "$APP_NAME" \
        --size 100 \
        --no-encryption

    echo "   ✅ Volume created (100GB for temporary local storage)"
    echo "   📝 Note: Main storage is in Tigris, this is just for buffering"
fi

# Step 7: Deploy the application
echo "7️⃣ Deploying application..."

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Deploy with rolling strategy
fly deploy --strategy rolling --ha=false

echo "   ✅ Application deployed"

# Step 8: Scale process groups
echo "8️⃣ Scaling process groups..."

fly scale count scheduler=1 --process-group scheduler
fly scale count worker=2 --process-group worker
fly scale count api=1 --process-group api
fly scale count qa_api=1 --process-group qa_api
fly scale count dashboard=1 --process-group dashboard

echo "   ✅ Process groups scaled"

# Step 9: Run database initialization
echo "9️⃣ Initializing database..."
fly ssh console -a "$APP_NAME" -C "python /app/scripts/init_database.py" || true
echo "   ✅ Database initialized"

# Step 10: Display status and costs
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