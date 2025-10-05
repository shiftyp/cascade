#!/bin/bash
# Manual migration runner for CASCADE Data Collector
# Usage: ./scripts/run_migrations.sh

set -e

# Get database URL from environment
DATABASE_URL="${DATABASE_URL:-postgresql://cascade:cascade@localhost/cascade_data}"

echo "CASCADE Database Migration Runner"
echo "=========================================="
echo "Database: $DATABASE_URL"
echo ""

# Parse DATABASE_URL to extract components
if [[ $DATABASE_URL =~ postgresql://([^:]+):([^@]+)@([^/:]+):?([0-9]*)/(.+) ]]; then
    PGUSER="${BASH_REMATCH[1]}"
    PGPASSWORD="${BASH_REMATCH[2]}"
    PGHOST="${BASH_REMATCH[3]}"
    PGPORT="${BASH_REMATCH[4]:-5432}"
    PGDATABASE="${BASH_REMATCH[5]}"
else
    echo "ERROR: Invalid DATABASE_URL format"
    echo "Expected: postgresql://user:pass@host:port/database"
    exit 1
fi

export PGPASSWORD

echo "Connecting to:"
echo "  Host: $PGHOST:$PGPORT"
echo "  Database: $PGDATABASE"
echo "  User: $PGUSER"
echo ""

# Test connection
echo "Testing database connection..."
if ! psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "SELECT 1" > /dev/null 2>&1; then
    echo "ERROR: Cannot connect to database"
    exit 1
fi
echo "✓ Connection successful"
echo ""

# Find migrations directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATIONS_DIR="$(dirname "$SCRIPT_DIR")/migrations"

if [ ! -d "$MIGRATIONS_DIR" ]; then
    echo "ERROR: Migrations directory not found: $MIGRATIONS_DIR"
    exit 1
fi

echo "Migrations directory: $MIGRATIONS_DIR"
echo ""

# Run each migration in order
echo "Running migrations..."
echo "=========================================="

for migration in $(ls "$MIGRATIONS_DIR"/*.sql | sort); do
    filename=$(basename "$migration")
    echo ""
    echo "► Running: $filename"

    if psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -f "$migration"; then
        echo "✓ Success: $filename"
    else
        echo "✗ Failed: $filename"
        echo ""
        echo "Migration failed. Continue anyway? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Stopping migrations."
            exit 1
        fi
    fi
done

echo ""
echo "=========================================="
echo "✓ All migrations completed!"
echo ""

# Show tables
echo "Current tables in database:"
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "\dt" 2>/dev/null || echo "Could not list tables"
