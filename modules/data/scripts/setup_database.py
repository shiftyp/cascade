#!/usr/bin/env python3
"""
Setup CASCADE database and run migrations
"""
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_config():
    """Get database configuration from environment or defaults"""
    return {
        'host': os.getenv('CASCADE_DB_HOST', os.getenv('DB_HOST', 'localhost')),
        'port': int(os.getenv('CASCADE_DB_PORT', '5432')),
        'database': os.getenv('CASCADE_DB_NAME', 'cascade_data'),
        'user': os.getenv('CASCADE_DB_USER', 'postgres'),
        'password': os.getenv('CASCADE_DB_PASSWORD', 'postgres')
    }

def create_database_if_not_exists(config):
    """Create database if it doesn't exist"""
    conn = None
    try:
        # Connect to postgres database
        conn_config = config.copy()
        conn_config['database'] = 'postgres'
        conn = psycopg2.connect(**conn_config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Check if database exists
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (config['database'],)
        )
        exists = cur.fetchone()

        if not exists:
            logger.info(f"Creating database '{config['database']}'...")
            cur.execute(f"CREATE DATABASE {config['database']}")
            logger.info("Database created successfully")
        else:
            logger.info(f"Database '{config['database']}' already exists")

        cur.close()
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()

def run_migrations(config):
    """Run all pending migrations"""
    conn = None
    try:
        conn = psycopg2.connect(**config)
        cur = conn.cursor()

        # Create migrations tracking table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # Get list of applied migrations
        cur.execute("SELECT version FROM schema_migrations")
        applied = {row[0] for row in cur.fetchall()}

        # Find migration files
        migrations_dir = Path(__file__).parent.parent / 'migrations'
        if not migrations_dir.exists():
            logger.error(f"Migrations directory not found: {migrations_dir}")
            sys.exit(1)

        migration_files = sorted(migrations_dir.glob('*.sql'))

        # Apply new migrations
        for migration_file in migration_files:
            version = migration_file.stem

            if version in applied:
                logger.info(f"Migration '{version}' already applied, skipping")
                continue

            logger.info(f"Applying migration '{version}'...")

            # Read and execute migration
            with open(migration_file, 'r') as f:
                sql = f.read()

            try:
                cur.execute(sql)

                # Record migration
                cur.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,)
                )

                conn.commit()
                logger.info(f"Migration '{version}' applied successfully")

            except Exception as e:
                conn.rollback()
                logger.error(f"Error applying migration '{version}': {e}")
                sys.exit(1)

        # Verify database structure
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]

        logger.info(f"Database tables: {', '.join(tables)}")

        # Check row counts
        for table in tables:
            if table != 'schema_migrations':
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                logger.info(f"  {table}: {count} rows")

        cur.close()
        logger.info("Database setup complete!")

    except Exception as e:
        logger.error(f"Error running migrations: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()

def main():
    """Main setup function"""
    logger.info("CASCADE Database Setup")
    logger.info("=" * 50)

    config = get_db_config()
    logger.info(f"Database: {config['host']}:{config['port']}/{config['database']}")

    # Create database if needed
    create_database_if_not_exists(config)

    # Run migrations
    run_migrations(config)

    logger.info("=" * 50)
    logger.info("Setup complete! Database is ready for use.")

if __name__ == "__main__":
    main()